"""Self-distillation step 5: LoRA on Gemma 4 31B from the packed self-distillation sequences.

Rank 8, attention layers only. Base = the weights the W4A16 serving model was made from
(google/gemma-4-31B-it-qat-q4_0-unquantized; its card lists it as base_model), loaded 4-bit nf4 (QLoRA) on one GPU.
Default targets are q_proj + o_proj of the 60 language layers: full-attention layers reuse k as v
(`attention_k_eq_v`), and a LoRA on k/v there has no proven serving path in vLLM — `--targets` can add them.

What the script does, all of it checked on CPU with a tiny random Gemma 4 (`--tiny`) before any GPU is rented:
  - loss only on the model's own output tokens (`loss_ranges` from tools/pack_examples.py), in 1,024-token chunks with
    the model's own logit softcap (30); token-weighted over each accumulation group;
  - hold-out runs (whole tasks, seeded) scored before training, after each epoch and at the end;
  - checkpoints every `--save-every` optimizer steps: adapter + optimizer + scheduler + step, written locally and
    copied to `--persist` (a durable disk) with a sha256 check, so a pre-empted machine resumes on a new one;
  - `--resume` continues from the newest checkpoint in --persist (or --out) with the same data order;
  - `--max-train-s` stops cleanly after a checkpoint; final adapter + manifest (shas, config, versions, losses).

    python tools/train_lora.py --base /path/base --data a.jsonl b.jsonl --out OUT --persist DURABLE/r1
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

CHUNK = 1024


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(out: Path, row: dict) -> None:
    row = {"t": round(time.time(), 1)} | row
    print(json.dumps(row), flush=True)
    with open(out / "train_log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def load_data(paths: list[str], max_len: int, holdout: int, seed: int) -> tuple[list, list, dict]:
    rows = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["n_tokens"] <= max_len and r["n_loss_tokens"] > 0:
                rows.append(r)
    tasks = sorted({r["task"] for r in rows})
    rng = random.Random(seed)
    held = set(rng.sample(tasks, min(holdout, len(tasks) // 5)))
    train = [r for r in rows if r["task"] not in held]
    evals = [r for r in rows if r["task"] in held]
    info = {"files": {p: sha256(Path(p)) for p in paths}, "sequences": len(rows), "tasks": len(tasks),
            "train_sequences": len(train), "holdout_sequences": len(evals), "holdout_tasks": sorted(held),
            "train_loss_tokens": sum(r["n_loss_tokens"] for r in train), "train_tokens": sum(r["n_tokens"] for r in train)}
    return train, evals, info


def tiny_base(out: Path, tok_dir: Path, real_config: Path) -> Path:
    """A tiny random Gemma 4 with the real architecture (layer pattern, k=v full attention, softcap, tied head)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = json.loads(real_config.read_text())
    cfg.pop("quantization_config", None)
    t = cfg["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
             num_global_key_value_heads=1, head_dim=32, global_head_dim=64, num_hidden_layers=6, sliding_window=64,
             layer_types=["sliding_attention"] * 5 + ["full_attention"])
    v = cfg["vision_config"]
    v.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
             head_dim=16, global_head_dim=16)
    d = out / "tiny_base"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(d), dtype=torch.float32)
    m.save_pretrained(d)
    for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(tok_dir / n, d / n)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="base model folder (bf16 safetensors)")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--persist", help="durable folder for checkpoints and the final adapter")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--targets", default="q_proj,o_proj")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--accum", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--holdout", type=int, default=8, help="whole tasks held out for the loss check")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--max-train-s", type=float, default=float("inf"))
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = no limit)")
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--no-offload", action="store_true", help="keep saved activations on the GPU")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ladder", help="memory/time ladder only: comma list of lengths, e.g. 4096,8192,16384")
    ap.add_argument("--tiny", action="store_true", help="CPU test: build a tiny random Gemma 4 (needs --tok-dir, --real-config)")
    ap.add_argument("--tok-dir")
    ap.add_argument("--real-config")
    ap.add_argument("--truncate", type=int, default=0, help="tests only: cut every sequence to this many tokens")
    a = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    import importlib.metadata as md

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    persist = Path(a.persist) if a.persist else None
    if persist:
        persist.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    offload = not a.no_offload
    base = tiny_base(out, Path(a.tok_dir), Path(a.real_config)) if a.tiny else Path(a.base)
    versions = {k: md.version(k) for k in ("torch", "transformers", "peft", "accelerate", "safetensors")}
    if cuda and a.quant == "nf4":
        versions["bitsandbytes"] = md.version("bitsandbytes")

    train, evals, info = load_data(a.data, a.max_len, a.holdout, a.seed)
    if a.truncate:                                          # tests only: a window around the first output span
        for r in train + evals:
            o = max(0, r["loss_ranges"][0][0] - a.truncate // 2)
            r["input_ids"] = r["input_ids"][o:o + a.truncate]
            r["loss_ranges"] = [[max(s - o, 1), min(e - o, a.truncate)] for s, e in r["loss_ranges"]
                                if s - o < a.truncate and e - o > 1]
    log(out, {"event": "data", **{k: v for k, v in info.items() if k != "holdout_tasks"}})

    # ---- model ----
    cfg = AutoConfig.from_pretrained(base)
    tcfg = getattr(cfg, "text_config", cfg)
    softcap = getattr(tcfg, "final_logit_softcapping", None)
    kw = dict(dtype=torch.bfloat16 if cuda else torch.float32, low_cpu_mem_usage=True)
    if cuda:
        kw |= dict(device_map={"": 0}, attn_implementation="sdpa")
        if a.quant == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    load_s = round(time.time() - t0, 1)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    tg = "|".join(re.escape(x) for x in a.targets.split(","))
    target_re = rf"^(?!.*vision)(?!.*audio).*\.layers\.\d+\.self_attn\.({tg})$"
    lc = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                    target_modules=target_re)
    torch.manual_seed(a.seed)                               # same LoRA init on every start (resume test compares bytes)
    pm = get_peft_model(model, lc)
    pm.train()          # gradient checkpointing only acts in train mode
    lora_mods = [n for n, _ in pm.named_modules() if n.endswith("lora_A")]
    n_layers = getattr(tcfg, "num_hidden_layers", 0)
    expect = n_layers * len(a.targets.split(","))
    log(out, {"event": "model", "load_s": load_s, "lora_modules": len(lora_mods), "expected": expect,
              "trainable": sum(p.numel() for p in pm.parameters() if p.requires_grad), "softcap": softcap,
              "device": str(dev), "quant": a.quant if cuda else "none(cpu)", "offload_activations": offload,
              "versions": versions})
    assert len(lora_mods) == expect, f"LoRA attached to {len(lora_mods)} modules, expected {expect}"

    inner = pm.get_base_model()
    body = inner.model
    head_w = inner.get_output_embeddings().weight          # tied to the embedding; frozen

    def seq_loss(r: dict) -> tuple[torch.Tensor, int]:
        ids = torch.tensor([r["input_ids"]], device=dev)
        n = ids.shape[1]
        tgt = torch.zeros(n, dtype=torch.bool)
        for s, e in r["loss_ranges"]:
            tgt[s:e] = True
        pos = torch.nonzero(tgt[1:]).squeeze(1)             # predict token p+1 from position p
        total = torch.zeros((), device=dev, dtype=torch.float32)

        def chunk(hc, yc):
            lg = (hc @ head_w.T.to(hc.dtype)).float()
            if softcap:
                lg = torch.tanh(lg / softcap) * softcap
            return F.cross_entropy(lg, yc, reduction="sum")
        # Memory on an A100 40GB: 4k 25.1 / 8k 28.8 GiB, 16k out of memory — the per-layer checkpoint inputs
        # (~0.9 GiB per 1k tokens) are what grows; they wait in pinned host RAM until the backward pass needs them
        with (torch.autograd.graph.save_on_cpu(pin_memory=cuda) if offload else contextlib.nullcontext()):
            h = body(input_ids=ids, use_cache=False).last_hidden_state[0]
            for i in range(0, len(pos), CHUNK):
                p = pos[i:i + CHUNK].to(dev)
                total = total + checkpoint(chunk, h[p], ids[0, p + 1], use_reentrant=False)
        return total, len(pos)

    @torch.no_grad()
    def eval_loss(rows) -> float:
        tot, n = 0.0, 0
        for r in rows:
            s, k = seq_loss(r)
            tot += float(s); n += k
        return round(tot / max(n, 1), 5)

    if a.ladder:
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr)
        longest = max(train, key=lambda r: r["n_tokens"])
        for L in [int(x) for x in a.ladder.split(",")]:
            r = {"input_ids": (longest["input_ids"] * (L // len(longest["input_ids"]) + 1))[:L], "loss_ranges": [[L // 2, L]]}
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t = time.time()
            try:
                s, k = seq_loss(r); (s / k).backward(); opt.step(); opt.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.synchronize()
                row = {"ok": True, "seconds": round(time.time() - t, 1)}
            except torch.cuda.OutOfMemoryError as e:
                row = {"ok": False, "oom": True, "error": str(e)[:200]}
            opt.zero_grad(set_to_none=True)
            row |= {"tokens": L, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None}
            log(out, {"event": "ladder", **row})
            if cuda:
                torch.cuda.empty_cache()
            if not row["ok"]:
                break
        return 0

    # ---- schedule (same order on every resume) ----
    rng = random.Random(a.seed)
    n_epochs = math.ceil(a.epochs)
    order = []
    for ep in range(n_epochs):
        idx = list(range(len(train)))
        rng.shuffle(idx)
        order += [(ep, i) for i in idx]
    order = order[:int(round(a.epochs * len(train)))]
    steps = math.ceil(len(order) / a.accum)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, int(a.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    def ckpt_dirs(root: Path | None):
        return sorted(root.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1])) if root and root.exists() else []

    start, resumed_losses = 0, {}
    if a.resume:
        cands = [d for d in ckpt_dirs(persist) + ckpt_dirs(out) if (d / "DONE").exists()]
        if cands:
            d = max(cands, key=lambda p: int(p.name.split("-")[1]))
            man = json.loads((d / "DONE").read_text())
            for f, h in man["sha256"].items():
                assert sha256(d / f) == h, f"checkpoint file {f} is damaged in {d}"
            from safetensors.torch import load_file
            sd = load_file(str(d / "adapter" / "adapter_model.safetensors"))
            from peft import set_peft_model_state_dict
            res = set_peft_model_state_dict(pm, sd)
            assert not getattr(res, "unexpected_keys", None), res
            st = torch.load(d / "state.pt", map_location=dev)
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start = st["step"]
            resumed_losses = man.get("losses") or {}
            log(out, {"event": "resumed", "from": str(d), "step": start})

    def save_ckpt(step: int, losses: dict) -> None:
        d = out / f"ckpt-{step}"
        if d.exists():
            shutil.rmtree(d)
        pm.save_pretrained(str(d / "adapter"))
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, d / "state.pt")
        files = [p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()]
        man = {"step": step, "sha256": {f: sha256(d / f) for f in files}, "losses": losses}
        (d / "DONE").write_text(json.dumps(man))
        if persist:                                         # copy, then verify from the copy (network FS: memory note)
            tmp = persist / f".tmp-ckpt-{step}"
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(d, tmp)
            for f, h in man["sha256"].items():
                assert sha256(tmp / f) == h, f"copy to {persist} damaged {f}"
            final = persist / d.name
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            for old in ckpt_dirs(persist)[:-2]:             # keep the newest 2
                shutil.rmtree(old)
        for old in ckpt_dirs(out)[:-2]:
            shutil.rmtree(old)
        log(out, {"event": "checkpoint", "step": step, "persisted": bool(persist)})

    losses = {"holdout_before": None} | resumed_losses
    if start == 0:
        pm.eval(); losses["holdout_before"] = eval_loss(evals); pm.train()
        log(out, {"event": "eval", "step": 0, "holdout_loss": losses["holdout_before"]})
    t_start = time.time()
    step = start
    per_epoch = math.ceil(len(train) / a.accum)
    while step < steps:
        group = order[step * a.accum:(step + 1) * a.accum]
        ts = time.time(); tot_tok, tot_loss, tot_len = 0, 0.0, 0
        n_loss = 0
        for _, i in group:                                   # token-weighted mean over the group
            tgt = sum(max(0, min(e, len(train[i]["input_ids"])) - max(s, 1)) for s, e in train[i]["loss_ranges"])
            n_loss += tgt
        for _, i in group:
            s, k = seq_loss(train[i])
            if k == 0:
                continue                                     # nothing to learn in this sequence (tests only)
            (s / max(n_loss, 1)).backward()
            tot_loss += float(s.detach()); tot_tok += k; tot_len += len(train[i]["input_ids"])
        gn = torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        row = {"event": "step", "step": step, "of": steps, "epoch": group[0][0], "loss": round(tot_loss / max(tot_tok, 1), 5),
               "loss_tokens": tot_tok, "tokens": tot_len, "grad_norm": round(float(gn), 4), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - ts, 1)}
        if cuda:
            row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log(out, row)
        end_epoch = step % per_epoch == 0 or step == steps
        if end_epoch:
            pm.eval(); hl = eval_loss(evals); pm.train()
            losses[f"holdout_step{step}"] = hl
            log(out, {"event": "eval", "step": step, "holdout_loss": hl})
        out_of_time = time.time() - t_start > a.max_train_s
        if step % a.save_every == 0 or end_epoch or out_of_time or step == steps:
            save_ckpt(step, losses)
        if out_of_time and step < steps:
            log(out, {"event": "stopped_on_time", "step": step})
            return 3

    # ---- final adapter ----
    fin = out / "adapter"
    if fin.exists():
        shutil.rmtree(fin)
    pm.save_pretrained(str(fin))
    from safetensors.torch import load_file
    keys = sorted(load_file(str(fin / "adapter_model.safetensors")))
    host_prefix = "base_model.model.model.language_model.layers."
    manifest = {"base": str(base), "base_config_sha256": sha256(Path(base) / "config.json"), "data": info,
                "lora": {"r": a.rank, "alpha": a.alpha, "targets": a.targets, "target_re": target_re},
                "train": {"lr": a.lr, "epochs": a.epochs, "accum": a.accum, "steps": steps, "seed": a.seed,
                          "max_len": a.max_len, "quant": a.quant}, "losses": losses, "versions": versions,
                "adapter_files": {p.name: sha256(p) for p in fin.iterdir()},
                "adapter_keys": len(keys), "host_key_format": all(k.startswith(host_prefix) for k in keys),
                "key_sample": keys[:2]}
    (fin / "c8_manifest.json").write_text(json.dumps(manifest, indent=1))
    if persist:
        dst = persist / "adapter"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(fin, dst)
        for f, h in manifest["adapter_files"].items():
            assert sha256(dst / f) == h, f"final adapter copy damaged: {f}"
    log(out, {"event": "done", "losses": losses, "host_key_format": manifest["host_key_format"], "adapter_keys": len(keys)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
