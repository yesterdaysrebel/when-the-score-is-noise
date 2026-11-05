"""Self-distillation step 6: does the organisers' vLLM apply our adapter to the W4A16 serving model, in the trained direction?

Run on the training box right after training (with the W4A16 serving model on local disk). Same engine settings as the organisers'
scorer for a bundle with an adapter (enable_lora, max_loras = 1, max_lora_rank = the
adapter's rank), the gemma4 model at revision ca9d5250, bf16, 32,768 context.
  1. LOAD   — the engine starts with LoRA on and accepts the adapter.
  2. ACTIVE + DIRECTION — on the held-out runs (tasks never trained on), the mean loss of the model's own output tokens
     (vLLM prompt_logprobs) is computed without and with the adapter. PASS = adapter lower than base. The training
     framework's hold-out loss (bf16/nf4 base) is printed beside it for comparison: same direction expected, not the
     same number (W4A16 differs from the nf4 training base).
  3. FORMAT — greedy continuations of held-out prompts with the adapter: every tool call parses
     (`<|tool_call>call:NAME{...}<tool_call|>` with NAME one of the 9 tools).
Writes serve_check.json; exit 0 only if 1-3 pass. No competition data is used (SWE-rebench hold-out runs only).

    python tools/serve_check.py --model-dir W4A16_DIR --adapter OUT/adapter --data a.jsonl b.jsonl --out OUT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch", "get_code_neighbors",
         "search_similar_code", "get_code_subgraph"}
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--adapter", required=True, help="folder with adapter_config.json + c8_manifest.json")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12, help="held-out sequences scored")
    ap.add_argument("--gen", type=int, default=6, help="held-out prompts continued greedily")
    a = ap.parse_args(argv)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    man = json.loads((Path(a.adapter) / "c8_manifest.json").read_text())
    held = set(man["data"]["holdout_tasks"])
    rows = [json.loads(l) for p in a.data for l in open(p)]
    rows = sorted((r for r in rows if r["task"] in held), key=lambda r: r["id"])[:a.n]
    assert rows, "no held-out sequences found"
    rank = man["lora"]["r"]
    rep = {"adapter_manifest": {k: man[k] for k in ("lora", "train", "losses", "adapter_files")}, "n_heldout": len(rows)}
    t = time.time()
    # A first run ran out of memory in prompt_logprobs (a 16,276-token prompt scored in 8,192-token chunks = 8 GiB of
    # full-vocabulary logits). Smaller prefill chunks + a little headroom; batching does not change the numbers.
    llm = LLM(model=a.model_dir, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.85, seed=20261021,
              enable_lora=True, max_loras=1, max_lora_rank=max(8, rank), enable_prefix_caching=False,
              max_num_batched_tokens=1024)
    rep["load_s"] = round(time.time() - t, 1)
    lora = LoRARequest("c8", 1, str(Path(a.adapter).resolve()))

    def nll(use_lora: bool) -> float:
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        tot, n = 0.0, 0
        for r in rows:
            o = llm.generate({"prompt_token_ids": r["input_ids"]}, sp, lora_request=lora if use_lora else None,
                             use_tqdm=False)[0]
            lp = o.prompt_logprobs                              # lp[i] = {token_id: Logprob} for position i (None at 0)
            for s, e in r["loss_ranges"]:
                for i in range(max(s, 1), e):
                    tot -= lp[i][r["input_ids"][i]].logprob; n += 1
        return round(tot / max(n, 1), 5)

    rep["load_ok"] = True
    rep["heldout_loss_base"] = nll(False)
    rep["heldout_loss_adapter"] = nll(True)
    rep["training_heldout_losses"] = man["losses"]
    rep["direction_ok"] = rep["heldout_loss_adapter"] < rep["heldout_loss_base"]

    gens, bad = [], 0
    sp = SamplingParams(max_tokens=1024, temperature=0.0, stop_token_ids=[1, 106, 50], skip_special_tokens=False)
    for r in rows[:a.gen]:
        s = r["loss_ranges"][-1][0]                             # continue from the start of the last output span
        o = llm.generate({"prompt_token_ids": r["input_ids"][:s]}, sp, lora_request=lora, use_tqdm=False)[0]
        txt = o.outputs[0].text
        calls = CALL.findall(txt)
        ok = all(n in TOOLS for n, _ in calls) and txt.count("<|tool_call>") == len(calls)
        bad += not ok
        gens.append({"id": r["id"], "calls": [n for n, _ in calls], "ok": ok, "finish": o.outputs[0].finish_reason,
                     "tokens": len(o.outputs[0].token_ids)})
    rep["generations"] = gens
    rep["format_ok"] = bad == 0
    rep["PASS"] = rep["load_ok"] and rep["direction_ok"] and rep["format_ok"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "serve_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("generations", "adapter_manifest")}, indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
