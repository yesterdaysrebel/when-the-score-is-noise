"""Self-distillation step 4: turn the kept runs into training sequences exactly as the served model saw them.

Rendering = what vLLM did at serve time, proven by token counts (all 3,540 prompts of the training runs match the
server's `usage.prompt_tokens` exactly; a first sample of 207 calls did too):
  - Gemma's chat template at HF revision ca9d5250 (the served one; hashes checked below), enable_thinking=True;
  - the tool list on every call (our body logger stored it only on a task's first call);
  - tool-call arguments as objects, message text as a list of text parts (vLLM's "openai" content format).

A run becomes one or more sequences. Calls are walked in order; call k's prompt P_k must start with the sequence so far
(previous prompt + previous output), else a new sequence starts (context compaction, or anything else that rewrites the
history). Each call adds its output A_k = the model's own tokens (thinking + tool call + the `<|tool_response>` stop
token, or thinking + final answer + `<turn|>`); only A_k tokens carry loss. The template drops thinking from a final
answer turn, so that one is built by hand. Every output is checked against `usage.completion_tokens`: 80% match exactly,
17% are +1 (the template always writes a newline before `<channel|>`; the model sometimes does not — kept, since the next
prompt holds the template's form anyway); an output more than 3 tokens off (a malformed call the parser rewrote) stays
in the sequence as context but carries no loss. Summary calls (compaction) are left out unless --with-summaries.

    python -I tools/pack_examples.py RUNS_DIR kept.json TOKENIZER_DIR --out pack/
"""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

GIT_BLOB_SHA1 = {"chat_template.jinja": "f62ca843a405a3066bdbddba8e42c92dd1ca4058",
                 "tokenizer_config.json": "af7f25861136b515f4bf64d9d0a6cf9875a6d508"}
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
SUMMARY_PREFIX: str | None = None    # first words of the harness's summarisation prompt (env HARNESS_SUMMARY_PREFIX)


def harness_string(var: str, what: str) -> str:
    """Harness-specific text is not shipped with this release: set it from your copy of the competition harness."""
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"set {var} to {what} (from the competition harness; not included in this release)")
    return v


def check_tokenizer(d: Path) -> None:
    for n, want in GIT_BLOB_SHA1.items():
        b = (d / n).read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == want, f"{n} is not the served revision"
    assert hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest() == TOKENIZER_SHA256, "tokenizer.json differs"


def read_bodies(path: Path, keep: set[str]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = collections.defaultdict(list)
    try:
        with gzip.open(path, "rt") as g:
            for line in g:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    break                                   # a worker died mid-line: everything before it is complete
                if d.get("task") in keep:
                    by[d["task"]].append(d)
    except EOFError:
        pass
    for t, lst in by.items():                               # a resumed task restarts seq: keep the last attempt
        lst.sort(key=lambda d: d["seq"]) if all(isinstance(d.get("seq"), int) for d in lst) else None
        starts = [i for i, d in enumerate(lst) if i == 0 or d["seq"] <= lst[i - 1]["seq"]]
        by[t] = lst[starts[-1]:]
    return by


def served_form(msgs: list[dict]) -> list[dict]:
    out = copy.deepcopy(msgs)
    for m in out:
        for tc in m.get("tool_calls") or []:
            a = tc.get("function", {}).get("arguments")
            if isinstance(a, str):
                try:
                    tc["function"]["arguments"] = json.loads(a)
                except json.JSONDecodeError:
                    pass
        if isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"]}]
    return out


def assistant_msg(resp: dict) -> dict:
    r = resp["choices"][0]["message"]
    m = {"role": "assistant", "content": r.get("content"), "reasoning_content": r.get("reasoning_content")}
    if r.get("tool_calls"):
        m["tool_calls"] = r["tool_calls"]
    return m


def is_summary(d: dict) -> bool:
    ms = d.get("messages") or []
    c = ms[0].get("content") if len(ms) == 1 else None
    global SUMMARY_PREFIX
    if SUMMARY_PREFIX is None:
        SUMMARY_PREFIX = harness_string("HARNESS_SUMMARY_PREFIX", "the first words of the harness's summarisation prompt")
    return isinstance(c, str) and c.startswith(SUMMARY_PREFIX)


class Packer:
    def __init__(self, tok):
        self.tok = tok
        self.st = collections.Counter()

    def render(self, msgs, tools, gen):
        return self.tok.apply_chat_template(served_form(msgs), tools=tools, add_generation_prompt=gen,
                                            enable_thinking=True, tokenize=False)

    def ntok(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def output_text(self, msgs, tools, resp, prompt):
        """A_k: the model's own text for this call, as it appears in the next prompt."""
        a = assistant_msg(resp)
        if not a.get("tool_calls"):
            # final answer: the template drops its thinking (and adds a newline after <turn|> that the model never
            # generates); the model wrote [thought channel] + answer + '<turn|>'
            think = a.get("reasoning_content") or ""
            return ("<|channel>thought\n" + think + "\n<channel|>" if think else "") + (a.get("content") or "") + "<turn|>"
        full = self.render(msgs + [a], tools, False)
        return full[len(prompt):] if full.startswith(prompt) else None      # None: counted as a failure by the caller

    def run(self, calls: list[dict], with_summaries: bool) -> list[dict]:
        seqs, cur, tools = [], None, None
        for d in calls:
            if d.get("tools"):
                tools = d["tools"]
            summ = is_summary(d)
            if summ and not with_summaries:
                self.st["summary_calls_skipped"] += 1
                continue
            t = None if summ else tools
            prompt = self.render(d["messages"], t, True)
            usage = d["response"].get("usage") or {}
            self.st["calls"] += 1
            self.st["prompt_count_match" if self.ntok(prompt) == usage.get("prompt_tokens") else "prompt_count_MISMATCH"] += 1
            out = self.output_text(d["messages"], t, d["response"], prompt)
            if out is None:
                self.st["output_unrenderable"] += 1
                cur = None
                continue
            diff = self.ntok(out) - (usage.get("completion_tokens") or -999)
            self.st[f"output_count_diff_{diff:+d}" if abs(diff) <= 3 else "output_count_diff_large"] += 1
            learn = abs(diff) <= 3
            if cur is not None and not summ and prompt.startswith(cur["text"]):
                cur["text"] = prompt
            else:
                if cur is not None:
                    seqs.append(cur)
                cur = {"text": prompt, "spans": []}
                self.st["sequences"] += 1
            if learn:
                cur["spans"].append((len(cur["text"]), len(cur["text"]) + len(out)))
            else:
                self.st["outputs_kept_as_context_only"] += 1
            cur["text"] += out
            if summ:
                seqs.append(cur)
                cur = None
        if cur is not None:
            seqs.append(cur)
        return seqs

    def tokenize(self, seq: dict) -> dict:
        enc = self.tok(seq["text"], add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        spans, ranges, start = seq["spans"], [], None
        for i, (a, b) in enumerate(offs):
            inside = any(s <= a < e for s, e in spans)
            if inside and start is None:
                start = i
            if not inside and start is not None:
                ranges.append([start, i]); start = None
        if start is not None:
            ranges.append([start, len(ids)])
        return {"input_ids": ids, "loss_ranges": ranges, "n_tokens": len(ids),
                "n_loss_tokens": sum(e - s for s, e in ranges)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("kept_json")
    ap.add_argument("tokenizer_dir")
    ap.add_argument("--out", required=True, help="output folder: examples.jsonl + stats.json")
    ap.add_argument("--with-summaries", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=16384, help="longer sequences are reported and left out")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    td = Path(a.tokenizer_dir)
    check_tokenizer(td)
    pk = Packer(AutoTokenizer.from_pretrained(td))
    kept = [r for r in json.loads(Path(a.kept_json).read_text())["runs"] if r["keep"]]
    by_worker = collections.defaultdict(set)
    for r in kept:
        by_worker[r["worker"]].add(r["task"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lengths, n_written, n_long = [], 0, 0
    with open(out / "examples.jsonl", "w") as f:
        for w, tasks in sorted(by_worker.items()):
            bodies = read_bodies(Path(a.run_root) / w / "bodies.jsonl.gz", tasks)
            for t in sorted(tasks):
                if not bodies.get(t):
                    pk.st["kept_run_without_bodies"] += 1
                    continue
                pk.st["runs"] += 1
                for i, s in enumerate(pk.run(bodies[t], a.with_summaries)):
                    e = pk.tokenize(s)
                    lengths.append(e["n_tokens"])
                    if e["n_tokens"] > a.max_tokens:
                        n_long += 1
                        continue
                    f.write(json.dumps({"id": f"{t}#{i}", "task": t, "worker": w, **e}) + "\n")
                    n_written += 1
    lengths.sort()
    stats = dict(sorted(pk.st.items())) | {
        "written": n_written, "left_out_over_max_tokens": n_long, "max_tokens": a.max_tokens,
        "tokens_total": sum(lengths), "len_median": lengths[len(lengths) // 2] if lengths else 0,
        "len_max": lengths[-1] if lengths else 0}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    bad = stats.get("prompt_count_MISMATCH", 0) + stats.get("output_unrenderable", 0)
    return 1 if bad > 0.01 * max(stats.get("calls", 1), 1) else 0


if __name__ == "__main__":
    sys.exit(main())
