"""Served-adapter direction check: does the adapter, as actually served (vLLM, quantized base + LoRA), predict the model's
own held-out output better than the base does? Standalone form of the notebook cell used on the organisers' stack.

Held-out positions come from tools/make_direction_probe.py (training hold-out tasks, sequences cut to <= 8k tokens). One
/v1/completions request per position and model: prompt = tokens[:p], max_tokens 1, top-20 logprobs -> log-probability of
the real token p. 120 s per request, 20 min total cap, stop after 5 failures in a row.
PASS = coverage >= 0.8 AND mean log-prob gain (adapter - base) > 0 with the sequence-clustered bootstrap 95% lower bound > 0.
Speed is recorded from 256-token generations (median of 3 per model).

Start the server with LoRA enabled and the adapter registered, e.g.
    vllm serve BASE --enable-lora --max-loras 1 --max-lora-rank 8 --lora-modules adapter=ADAPTER_DIR
    python tools/direction_check.py --probe probe.json.gz --adapter adapter --out direction_check.json
Exit 0 only on PASS.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import urllib.request


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="gzip'd JSON from tools/make_direction_probe.py")
    ap.add_argument("--adapter", required=True, help="served model name of the adapter (vLLM --lora-modules NAME=...)")
    ap.add_argument("--base", help="served model name of the base (default: the first non-adapter model)")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="direction_check.json")
    ap.add_argument("--time-cap-s", type=int, default=20 * 60)
    a = ap.parse_args(argv)

    def api(path, body=None, timeout=420):
        req = urllib.request.Request(a.server + path, data=(json.dumps(body).encode() if body is not None else None),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    probe = json.loads(gzip.decompress(open(a.probe, "rb").read()))
    base_id = a.base or next(m["id"] for m in api("/v1/models")["data"] if m["id"] != a.adapter)

    def lp(model, ids, p):
        r = api("/v1/completions", {"model": model, "prompt": ids[:p], "max_tokens": 1, "temperature": 0, "logprobs": 20,
                                    "return_tokens_as_token_ids": True}, timeout=120)
        return r["choices"][0]["logprobs"]["top_logprobs"][0].get(f"token_id:{ids[p]}")

    dirc = {"base": base_id, "adapter": a.adapter, "positions": len(probe["pos"]), "rows": [], "aborted": None}
    t0, fails = time.time(), 0
    for si, p in probe["pos"]:
        if time.time() - t0 > a.time_cap_s:
            dirc["aborted"] = f"time cap {a.time_cap_s} s"; break
        ids = probe["seqs"][si]
        try:
            dirc["rows"].append([si, p, lp(base_id, ids, p), lp(a.adapter, ids, p)]); fails = 0
        except Exception as e:  # noqa: BLE001 — a failed request is counted, not fatal
            fails += 1; dirc.setdefault("errors", []).append(repr(e)[:200])
            if fails >= 5:
                dirc["aborted"] = "5 failures in a row"; break
    both = [(si, b, ad) for si, p, b, ad in dirc["rows"] if ad is not None and b is not None]
    pairs = [(si, ad - b) for si, b, ad in both]
    dirc["coverage"] = round(len(pairs) / max(1, dirc["positions"]), 3)
    dirc["seconds"] = round(time.time() - t0)
    if pairs:
        dirc["mean_gain"] = sum(d for _, d in pairs) / len(pairs)
        dirc["base_mean_logprob"] = sum(b for _, b, _ in both) / len(both)
        dirc["adapter_mean_logprob"] = sum(ad for _, _, ad in both) / len(both)
        by = {}
        for si, d in pairs:
            by.setdefault(si, []).append(d)
        ks, r, boot = sorted(by), random.Random(7), []
        for _ in range(2000):                     # resample whole sequences: positions in one sequence are not independent
            v = [d for k in (r.choice(ks) for _ in ks) for d in by[k]]
            boot.append(sum(v) / len(v))
        boot.sort(); dirc["gain_ci95"] = [boot[49], boot[1949]]
    dirc["pass"] = bool(pairs) and dirc["coverage"] >= 0.8 and dirc.get("gain_ci95", [0])[0] > 0

    def speed(model):
        ids = probe["seqs"][0][:4096]; t = time.time()
        r = api("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 256, "min_tokens": 256, "temperature": 0}, timeout=300)
        return round(r["usage"]["completion_tokens"] / (time.time() - t), 2)
    try:
        dirc["tok_per_s"] = {"base": sorted(speed(base_id) for _ in range(3))[1],
                             "adapter": sorted(speed(a.adapter) for _ in range(3))[1]}
        dirc["adapter_speed_ratio"] = round(dirc["tok_per_s"]["adapter"] / dirc["tok_per_s"]["base"], 3)
    except Exception as e:  # noqa: BLE001
        dirc["speed_error"] = repr(e)[:200]
    with open(a.out, "w") as f:
        json.dump(dirc, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in dirc.items() if k != "rows"}, indent=1, default=str))
    return 0 if dirc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
