"""§3 — how noisy is the score? F1 per-task outcome matrix, F2 power curve, noise_numbers.json.

Inputs: derived/outcomes.json — four 60-task runs of the same stack (Kaggle 4xL4, the organisers' harness, Gemma-4-31B
W4A16, one fixed server seed and task order): the base agent twice and the self-distilled adapter (§5.1) twice — and
derived/hidden_scores.json (leaderboard draws of the unchanged base).

Noise model. All four runs are treated as draws of one agent (an assumption: §5.1 detects no change in solves, which
does not confirm it; the base pair's own flip count is reported beside it). Each task is solved independently with probability p_i. The p_i come from an
empirical-Bayes beta-binomial fit to the per-task solve counts: Beta(a, b) is fitted by maximum likelihood to the counts,
and p_i is its posterior mean (k_i + a) / (R + a + b). Tasks never or always solved keep a small, data-fitted chance of
flipping instead of a fixed prior's (a Beta(1, 1) prior would move them to 1/6 and 5/6 and inflate the noise). Whole
runs may also vary together (the same-seed adapter pair differed by more than the per-task model predicts), so the power
analysis is also given under a run-level model, where a run's solve count is normal with the spread of the four
observed runs. Four runs cannot pin that spread down: it is a sensitivity bound, not an estimate. Both power analyses
are best cases: a true gain is placed on the least-solved tasks (raising them towards certainty, the lowest-variance
kind of gain), the critical value comes from the model's own null, and the run-level SD is treated as known.

    python analysis/noise.py        # writes figures/{f1_outcomes,f2_power}.png and figures/noise_numbers.json
"""
from __future__ import annotations

import itertools
import json
from math import erf, lgamma, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DERIVED, OUT = HERE / "derived", HERE / "figures"
EFFECTS, NS = [2, 4, 6, 8], [1, 2, 3, 5, 10]


def load():
    d = json.loads((DERIVED / "outcomes.json").read_text())
    h = json.loads((DERIVED / "hidden_scores.json").read_text())
    return d["runs"], np.array(d["matrix"], dtype=int), h["base"]          # runs x tasks


def fit_beta_binomial(k: np.ndarray, n: int) -> tuple[float, float]:
    """Maximum-likelihood Beta(a, b) for per-task counts k out of n (grid search on log a, log b)."""
    def ll(a, b):
        return sum(lgamma(x + a) + lgamma(n - x + b) - lgamma(n + a + b) + lgamma(a + b) - lgamma(a) - lgamma(b) for x in k)
    grid = np.exp(np.linspace(-5, 3, 161))
    return max(((ll(a, b), float(a), float(b)) for a in grid for b in grid))[1:]


def power(p, n_runs, effects, sims=4000, seed=7):
    """P(detect) of a true +effect tasks (spread over the least-solved tasks) with n runs per arm, two-sided 5%."""
    rng = np.random.default_rng(seed)
    out = {}
    null = np.array([rng.binomial(1, np.tile(p, (n_runs, 1))).sum(1).mean() - rng.binomial(1, np.tile(p, (n_runs, 1))).sum(1).mean()
                     for _ in range(sims)])
    crit = np.quantile(np.abs(null), 0.95)
    for e in effects:
        q = p.copy()
        need = e
        for i in np.argsort(p):                       # gains land on tasks the agent usually misses
            add = min(1 - q[i], need); q[i] += add; need -= add
            if need <= 1e-9:
                break
        diff = np.array([rng.binomial(1, np.tile(q, (n_runs, 1))).sum(1).mean() - rng.binomial(1, np.tile(p, (n_runs, 1))).sum(1).mean()
                         for _ in range(sims)])
        out[e] = float((diff > crit).mean())
    return float(crit), out


def power_run_level(sd, n_runs, effect):
    """Normal approximation: run solve counts ~ N(mu, sd); difference of arm means, two-sided 5%."""
    z = 1.959964 - effect / (sd * sqrt(2 / n_runs))
    return 1 - 0.5 * (1 + erf(z / sqrt(2)))


def compute(runs: list[str], M: np.ndarray, hidden: list[float]) -> dict:
    R, T = M.shape
    k = M.sum(0); solved = M.sum(1)
    a, b = fit_beta_binomial(k, R)
    p = (k + a) / (R + a + b)
    curves = {}
    for n in NS:
        crit, pw = power(p, n, EFFECTS)
        curves[n] = {"critical_diff": round(crit, 2), "power": {str(e): round(v, 2) for e, v in pw.items()}}
    sd_runs = float(solved.std(ddof=1))
    rng = np.random.default_rng(11)
    sim = rng.binomial(1, p, size=(100000, 2, T)).sum(2)     # two independent same-agent runs
    gap = abs(int(solved[2]) - int(solved[3]))
    p_gap = float((np.abs(sim[:, 0] - sim[:, 1]) >= gap).mean())
    sim4 = rng.binomial(1, p, size=(50000, R, T)).sum(2)              # R independent runs: largest of all pairwise gaps
    p_max_gap = float(((sim4.max(1) - sim4.min(1)) >= gap).mean())
    base, adapter = [i for i, r in enumerate(runs) if r.startswith("base")], [i for i, r in enumerate(runs) if r.startswith("adapter")]
    d = M[adapter].mean(0) - M[base].mean(0)
    bs = np.random.default_rng(3).choice(d, size=(20000, T)).sum(1)
    h = np.array(hidden)
    return {
        "runs": runs, "solved": solved.tolist(), "tasks": T,
        "flips": {f"{runs[i]} vs {runs[j]}": int((M[i] != M[j]).sum()) for i, j in itertools.combinations(range(R), 2)},
        "tasks_always_solved": int((k == R).sum()), "never_solved": int((k == 0).sum()),
        "sometimes": int(((k > 0) & (k < R)).sum()),
        "per_task_model": {"beta_a": round(a, 3), "beta_b": round(b, 3),
                           "sd_of_one_run_solves": round(float(np.sqrt((p * (1 - p)).sum())), 2),
                           "sd_raw_rates": round(float(np.sqrt(((k / R) * (1 - k / R)).sum())), 2),
                           "p_same_seed_gap_at_least_adapter_pair": round(p_gap, 4),
                           "p_largest_of_all_pairwise_gaps_at_least_that": round(p_max_gap, 3)},
        "sd_of_observed_run_solves": round(sd_runs, 2),
        "power_by_runs_per_arm": curves,
        "power_run_level_gain4": {str(n): round(power_run_level(sd_runs, n, 4), 2) for n in NS},
        "adapter_minus_base": {"solves": round(float(d.sum()), 2),
                               "task_paired_bootstrap_ci95": [round(float(x), 1) for x in np.percentile(bs, [2.5, 97.5])]},
        "hidden_base": {"draws": len(h), "mean": round(float(h.mean()), 3), "sd": round(float(h.std(ddof=1)), 3),
                        "sd_tasks_of_60": round(float(h.std(ddof=1) * 60), 1), "min": float(h.min()), "max": float(h.max())},
    }


def plot(nums: dict, M: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, T = M.shape
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.imshow(M[:, np.argsort(-M.sum(0), kind="stable")], aspect="auto", cmap="Greys", vmin=0, vmax=1.4)
    ax.set_yticks(range(R)); ax.set_yticklabels([f"{k} ({s}/{T})" for k, s in zip(nums["runs"], nums["solved"])])
    ax.set_xticks([]); ax.set_xlabel(f"{T} tasks, sorted by how often solved — {nums['tasks_always_solved']} always, "
                                     f"{nums['sometimes']} sometimes, {nums['never_solved']} never")
    ax.set_title("Same stack, same seed: which tasks were solved (dark = solved)")
    fig.tight_layout(); fig.savefig(out / "f1_outcomes.png", dpi=160); plt.close(fig)

    curves = nums["power_by_runs_per_arm"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for e in EFFECTS:
        ax.plot(NS, [curves[n]["power"][str(e)] for n in NS], marker="o", label=f"true gain +{e} tasks")
    ax.plot(NS, [nums["power_run_level_gain4"][str(n)] for n in NS], ls="--", c="#ff7f0e", marker="x",
            label=f"+4, runs vary as observed (SD {nums['sd_of_observed_run_solves']})")
    ax.axhline(0.8, ls=":", c="grey"); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"runs per arm ({T} tasks each)"); ax.set_ylabel("chance of detecting the gain (5% test)")
    ax.set_title(f"How many runs a {T}-task comparison needs"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "f2_power.png", dpi=160); plt.close(fig)


def main() -> int:
    runs, M, hidden = load()
    nums = compute(runs, M, hidden)
    OUT.mkdir(exist_ok=True)
    (OUT / "noise_numbers.json").write_text(json.dumps(nums, indent=1))
    plot(nums, M, OUT)
    print(json.dumps(nums, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
