"""Self-distillation: positions for the Kaggle direction check — does the served adapter predict Gemma's own held-out output better?

From the packed sequences of the training hold-out tasks (whole tasks never trained on; c8_manifest.json), each cut to
at most --max-len tokens (long scoring requests can run out of memory). --per-seq positions are
drawn (seeded) inside the model's own output spans. In the notebook, each position p is one /v1/completions request:
prompt = tokens[:p], max_tokens 1, top-20 logprobs → log-probability of the real token p, base vs adapter.
Output: gzip'd JSON (SWE-rebench content, CC-BY-4.0).

    python tools/make_direction_probe.py --manifest c8_manifest.json --data a.jsonl b.jsonl --out probe.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="the adapter's c8_manifest.json (holdout_tasks)")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--per-seq", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args(argv)
    held = set(json.loads(Path(a.manifest).read_text())["data"]["holdout_tasks"])
    rows = sorted((json.loads(l) for p in a.data for l in open(p)), key=lambda r: r["id"])
    rows = [r for r in rows if r["task"] in held]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    seqs, pos, tasks = [], [], []
    for r in rows:
        if len(seqs) == a.max_seqs:
            break
        ids = r["input_ids"][:a.max_len]
        cand = [p for s, e in r["loss_ranges"] for p in range(max(s, 1), min(e, len(ids)))]
        if len(cand) < a.per_seq:
            continue
        si = len(seqs)
        seqs.append(ids)
        tasks.append(r["id"])
        pos += [[si, p] for p in sorted(rng.sample(cand, a.per_seq))]     # ascending: prefix cache reuse in the server
    body = {"seqs": seqs, "pos": pos, "seq_ids": tasks, "max_len": a.max_len, "seed": a.seed}
    raw = json.dumps(body, separators=(",", ":")).encode()
    Path(a.out).write_bytes(gzip.compress(raw, 9))
    print(json.dumps({"seqs": len(seqs), "positions": len(pos), "tasks": len({t.split('#')[0] for t in tasks}),
                      "tokens": sum(map(len, seqs)), "gz_bytes": Path(a.out).stat().st_size,
                      "sha256": hashlib.sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
