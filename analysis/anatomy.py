"""§4 — where the budget goes. F3 time anatomy, F4 trajectory anatomy, anatomy_numbers.json.

Inputs: derived/anatomy_task_runs.json (one row per task-run of the two base runs, 120 task-runs: outcome, calls, first
edit, compaction, re-reads, generated characters by kind) and derived/timing_requests.json (latency and completion tokens
of every successful model request; the adapter runs give the adapter-overhead bar). Groups: solved; unsolved that used
(almost) the whole budget (a cap hit, or <= 2 of 28 calls left); unsolved that stopped early.

Definitions (the mechanism metrics of §6): first edit = 1-based index of the first edit_file/write_file call (calls + 1 if
none); compaction = the harness replaced the history with a summary at least once; true re-read = a read_file whose line
range is >= 50% already read in this task-run; "after a compaction" = every overlapping earlier read happened before the
latest compaction.

    python analysis/anatomy.py      # writes figures/{f3_time,f4_trajectory}.png and figures/anatomy_numbers.json
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]
DERIVED, OUT = HERE / "derived", HERE / "figures"
BUDGET_CALLS = 28
ORDER = ["solved", "unsolved: used budget", "unsolved: stopped early"]


def group(r):
    if r["resolved"]:
        return "solved"
    return "unsolved: used budget" if (r["hit_cap"] or BUDGET_CALLS - r["tool_calls"] <= 2) else "unsolved: stopped early"


def main() -> int:
    tr = json.loads((DERIVED / "anatomy_task_runs.json").read_text())
    base, rows = tr["runs"], tr["rows"]
    reqs = json.loads((DERIVED / "timing_requests.json").read_text())["rows"]
    nums = {}

    # ---- time anatomy ----
    rates = defaultdict(list)
    for q in reqs:
        rates[q["run"]].append(q["completion_tokens"] / q["latency_s"])
    t_model = sum(q["latency_s"] for q in reqs if q["run"] in base)
    t_task = sum(r["duration_s"] for r in rows if r["has_result"])
    R = sum(r["chars"]["reasoning"] for r in rows); C = sum(r["chars"]["content"] for r in rows)
    A = sum(r["chars"]["tool_args"] for r in rows)
    tps = {k: st.median(v) for k, v in rates.items()}
    adapter = [k for k in tps if k not in base]
    nums["time"] = {"model_share": round(t_model / t_task, 3), "reasoning_share_of_generated_chars": round(R / (R + C + A), 3),
                    "tool_args_share": round(A / (R + C + A), 3),
                    "tok_per_s_base": round(sum(tps[k] for k in base) / len(base), 1),
                    "tok_per_s_adapter": round(sum(tps[k] for k in adapter) / len(adapter), 1)}

    # ---- trajectory anatomy ----
    G = defaultdict(list)
    for r in rows:
        if r["has_result"]:
            G[group(r)].append(r)
    nums["trajectory"] = {k: {"runs": len(G[k]), "median_first_edit_call": st.median(r["first_edit_call"] for r in G[k]),
                              "mean_calls": round(st.mean(r["trace_calls"] for r in G[k]), 1),
                              "compacted_share": round(sum(r["compacted"] for r in G[k]) / len(G[k]), 2),
                              "true_rereads_per_run": round(sum(r["true_rereads"] for r in G[k]) / len(G[k]), 1),
                              "rereads_after_compaction_share": round(sum(r["rereads_after_compaction"] for r in G[k])
                                                                      / max(1, sum(r["true_rereads"] for r in G[k])), 2)}
                          for k in ORDER}
    # ---- numbers quoted in §4's text (computed from the same derived rows) ----
    base_tok = sorted(q["completion_tokens"] for q in reqs if q["run"] in base)
    done = [r for r in rows if r["has_result"]]
    mean_task_s = sum(r["duration_s"] for r in done) / len(done)
    nums["text_numbers"] = {
        "completion_tokens_per_call_median": st.median(base_tok),
        "completion_tokens_per_call_p99": base_tok[int(0.99 * (len(base_tok) - 1))],
        "completion_tokens_per_task": round(sum(base_tok) / len(done)),
        "mean_minutes_per_task": round(mean_task_s / 60, 2),
        "hours_per_60_task_run": round(mean_task_s * 60 / 3600, 1),
        "rereads_after_compaction_per_run": {k: round(T_["true_rereads_per_run"] * T_["rereads_after_compaction_share"], 2)
                                             for k, T_ in nums["trajectory"].items()},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "anatomy_numbers.json").write_text(json.dumps(nums, indent=1))

    # F3: time anatomy
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ms = nums["time"]["model_share"]; rs = nums["time"]["reasoning_share_of_generated_chars"]
    ax[0].barh(["task time"], [ms * rs], color="#4c72b0", label="model: reasoning")
    ax[0].barh(["task time"], [ms * (1 - rs)], left=[ms * rs], color="#55a868", label="model: text + tool arguments")
    ax[0].barh(["task time"], [1 - ms], left=[ms], color="#c44e52", label="tools + sandbox")
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("share of task time (model time split by generated characters)")
    ax[0].set_yticks([]); ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=0.9)
    n_runs = sum(len(v) for v in G.values())
    ax[0].set_ylim(-0.6, 1.0); ax[0].set_title(f"Where a task's time goes (base agent, {n_runs} runs)")
    bars = [nums["time"]["tok_per_s_base"], nums["time"]["tok_per_s_adapter"]]
    ax[1].bar(["base", "with LoRA adapter"], bars, color=["#4c72b0", "#dd8452"])
    ax[1].set_ylabel("tokens / s"); ax[1].set_title("Generation speed (median per call)"); ax[1].set_ylim(0, max(bars) * 1.2)
    for i, v in enumerate(bars):
        ax[1].text(i, v + 0.5, f"{v}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "f3_time.png", dpi=160); plt.close(fig)

    # F4: trajectory anatomy
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    lab = ["solved", "unsolved:\nused budget", "unsolved:\nstopped early"]; col = ["#55a868", "#c44e52", "#8172b2"]
    T = nums["trajectory"]
    ax[0].bar(lab, [T[k]["median_first_edit_call"] for k in ORDER], color=col); ax[0].set_title("First edit at call (median)")
    ax[1].bar(lab, [T[k]["compacted_share"] for k in ORDER], color=col); ax[1].set_ylim(0, 1); ax[1].set_title("Runs with history compacted")
    rr = [T[k]["true_rereads_per_run"] for k in ORDER]; ac = [T[k]["rereads_after_compaction_share"] for k in ORDER]
    ax[2].bar(lab, [r * a for r, a in zip(rr, ac)], color=col)
    ax[2].bar(lab, [r * (1 - a) for r, a in zip(rr, ac)], bottom=[r * a for r, a in zip(rr, ac)], color=col, alpha=0.4)
    ax[2].set_title("Re-reads of seen lines, per run"); ax[2].set_ylim(0, max(rr) * 1.4)        # headroom keeps the legend off the bars
    ax[2].legend(handles=[Patch(color="grey"), Patch(color="grey", alpha=0.4)],
                 labels=["after a compaction", "no compaction between"], fontsize=7, loc="upper left")
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "f4_trajectory.png", dpi=160); plt.close(fig)
    print(json.dumps(nums, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
