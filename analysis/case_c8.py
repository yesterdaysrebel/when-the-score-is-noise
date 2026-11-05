"""§5.1 — C8 self-distillation: it learns, the learning reaches the served model, the solve count does not move.

Inputs: derived/case_c8.json (QLoRA training log; direction check of the served adapter on the organisers' stack — vLLM
0.19.1, W4A16 base + adapter, 320 held-out positions), derived/outcomes.json (local solves) and derived/hidden_scores.json
(leaderboard draws).

    python analysis/case_c8.py      # writes figures/f5_case_c8.png and figures/case_c8_numbers.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
DERIVED, OUT = HERE / "derived", HERE / "figures"


def main() -> int:
    c8 = json.loads((DERIVED / "case_c8.json").read_text())
    oc = json.loads((DERIVED / "outcomes.json").read_text())
    hidden = json.loads((DERIVED / "hidden_scores.json").read_text())
    steps = [r for r in c8["train_log"] if r["event"] == "step"]
    evals = [r for r in c8["train_log"] if r["event"] == "eval"]
    d = c8["direction_check"]
    local = {k: sum(row) for k, row in zip(oc["runs"], oc["matrix"])}
    T = oc["tasks"]
    hb, hc = hidden["base"], hidden["c8"]
    nums = {"holdout_loss": {f"step {e['step']}": e["holdout_loss"] for e in evals},
            "served_mean_logprob": {"base": round(d["base_mean_logprob"], 3), "adapter": round(d["adapter_mean_logprob"], 3),
                                    "gain": round(d["mean_gain"], 3), "ci95": [round(x, 3) for x in d["gain_ci95"]],
                                    "positions": d["positions"], "coverage": d["coverage"]},
            "tok_per_s": d.get("tok_per_s"), f"local_solved_of_{T}": local, "hidden": {"base": hb, "c8": hc}}
    OUT.mkdir(exist_ok=True)
    (OUT / "case_c8_numbers.json").write_text(json.dumps(nums, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    if steps:   # per-step training loss is only in the full training log (absent when built from the final manifest)
        ax[0].plot([s["step"] for s in steps], [s["loss"] for s in steps], lw=1, c="#4c72b0", label="training (per step)")
    ax[0].plot([e["step"] for e in evals], [e["holdout_loss"] for e in evals], "o-", c="#c44e52", label="held-out tasks")
    for e in evals:
        ax[0].annotate(f"{e['holdout_loss']:.3f}", (e["step"], e["holdout_loss"]), textcoords="offset points", xytext=(6, 6),
                       ha="left", fontsize=7)
    ax[0].margins(x=0.18, y=0.15)
    ax[0].set_xlabel("optimizer step (2 epochs)"); ax[0].set_ylabel("loss (own output tokens)")
    ax[0].set_title("1. It learns"); ax[0].legend(fontsize=8)

    b, a = d["base_mean_logprob"], d["adapter_mean_logprob"]; lo, hi = d["gain_ci95"]
    ax[1].bar(["base", "adapter"], [b, a], color=["#4c72b0", "#dd8452"])
    ax[1].set_xlabel(f"gain +{a - b:.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (8 held-out sequences, {d['positions']} positions)", fontsize=7)
    ax[1].set_ylabel("mean log-prob (held-out)")
    ax[1].set_title("2. The served 4-bit model\nshows it (Kaggle, organisers' vLLM)", fontsize=10)

    xs = ["base A", "base B", "adapter A", "adapter B"]
    ax[2].bar(xs, [local[k] for k in xs], color=["#4c72b0", "#4c72b0", "#dd8452", "#dd8452"])
    ax[2].set_ylabel(f"solved of {T} (local)"); ax[2].set_ylim(0, 34)
    ax[2].text(0.5, 29.5, f"hidden: {len(hb)} draws\n{min(hb):.2f}–{max(hb):.2f}, mean {sum(hb)/len(hb):.3f}", ha="center", fontsize=7)
    ax[2].text(2.5, 29.5, f"hidden: {len(hc)} draw{'s' * (len(hc) > 1)}\n" + ", ".join(f"{x:.2f}" for x in hc), ha="center", fontsize=7)
    ax[2].set_title("3. No detectable change in solves", fontsize=10); ax[2].tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "f5_case_c8.png", dpi=160); plt.close(fig)
    print(json.dumps(nums, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
