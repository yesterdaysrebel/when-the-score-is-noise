# When the Score Is Noise — code and data

Companion release for the paper *When the Score Is Noise: Mechanism Metrics for Evaluating Budget-Limited Coding Agents*
(Gemma 4 Developer Agent paper track). It regenerates every figure in the paper and the numbers in
`figures/*_numbers.json`, and holds the tools behind the self-distillation case study. Some counts in the paper's text
(for example the 26 s spent after the first submit, the self-distillation run counts and Table 1) come from logs
that are not released; the paper says so. "C8" in file names is our label for the self-distillation experiment (§5.1).

Licences: the code and the derived JSON are Apache-2.0 (see `LICENSE`). The adapter is a Gemma derivative and is subject
to the Gemma Terms of Use (see `adapter/NOTICE`). `data/audit.json` keeps its source licence (below).

## What is here

| path | what |
|---|---|
| `analysis/noise.py` | §3: per-task outcome matrix (F1), power curve (F2), flips, noise models, the adapter-vs-base interval |
| `analysis/anatomy.py` | §4: time anatomy (F3), trajectory anatomy (F4) |
| `analysis/case_c8.py` | §5.1: self-distillation case (F5) |
| `derived/` | the only inputs of `analysis/`: anonymised per-task and per-request numbers (see below) |
| `figures/` | output of `analysis/` (PNG + `*_numbers.json`, the numbers quoted in the paper) |
| `tools/filter_runs.py` | picks which solved agent runs become training examples (clean-run rules v3) |
| `tools/pack_examples.py` | renders runs into training sequences exactly as the server tokenised them (checked against the server's token counts) |
| `tools/train_lora.py` | QLoRA trainer for Gemma 4 31B, loss on the model's own output tokens only, resumable checkpoints |
| `tools/serve_check.py` | does vLLM load the adapter on the quantized serving model, and move it in the trained direction? |
| `tools/make_direction_probe.py`, `tools/direction_check.py` | the served-adapter direction check (held-out log-prob, base vs adapter, sequence-clustered bootstrap) |
| `data/audit.json` | the audit (by a separate read-only LLM agent) of 20 kept and 3 dropped training runs, with the filter rules it led to |
| `adapter/` | the trained LoRA adapter (rank 8, q_proj + o_proj, bf16, 240 tensors), its PEFT config and `c8_manifest.json` (training setup, held-out tasks, losses, file hashes; needed by `tools/serve_check.py` and `tools/make_direction_probe.py`) |

## Reproduce the figures

```
pip install matplotlib numpy      # Python 3.9 or newer
python analysis/noise.py
python analysis/anatomy.py
python analysis/case_c8.py
```

Each script reads `derived/` only and writes into `figures/`. Nothing needs a GPU or network access.

## The tools

`tools/` are reference implementations, not a runnable pipeline: their inputs are our agent-run logs (request and
response bodies, `runs.jsonl`, patches), which are not released, and training needs torch, transformers, peft and
bitsandbytes; the direction check needs vLLM. Three harness-specific strings are not shipped either: set
`HARNESS_SUMMARY_PREFIX`, `HARNESS_REFUSED_TEXT` and `HARNESS_BUDGET_MARKER` from your copy of the competition harness.

`adapter/adapter_config.json` names the served W4A16 build as its base, so the organisers' vLLM loads it directly. The
adapter was trained on a 4-bit nf4 load of the unquantised QAT checkpoint (`c8_manifest.json`); the direction check
exists because those two bases differ.

## What `derived/` holds

- `outcomes.json`: four 60-task runs on the competition stack (Kaggle 4×L4, the organisers' harness, Gemma-4-31B W4A16,
  one fixed server seed and task order). `matrix[run][task]` is 1 if the task's tests passed. Tasks are indices into
  that fixed order.
- `anatomy_task_runs.json`: one row per task-run of the two base runs. Each row has the outcome, the call count, the
  index of the first edit, whether the history was compacted, re-read counts, and the number of generated characters by
  kind (reasoning, text, tool arguments).
- `timing_requests.json`: latency and completion tokens of every successful model request.
- `case_c8.json`: held-out loss before training and after each epoch, and the aggregates of the direction check on
  the served model.
- `hidden_scores.json`: our leaderboard scores (the fraction of hidden tasks solved).

## Noise model (`analysis/noise.py`)

Each task is solved independently with a probability fitted by empirical Bayes (a beta-binomial fit to the per-task
solve counts of the four runs). This is a lower bound on the noise, because whole runs can also vary together, so the
power curve for a four-task gain is also given under a run-level model, where a run's solve count is normal with the
spread of the four observed runs. The adapter-vs-base interval is a task-paired bootstrap over the 60 tasks.

## Mechanism metrics

All are per task-run, computed from the agent's tool-call trace and the model request log:

- **first edit**: 1-based index of the first `edit_file`/`write_file` call (calls + 1 if there is none).
- **compaction**: the harness replaced the history with a summary at least once.
- **true re-read**: a `read_file` whose requested line range was already ≥ 50% read earlier in the same task-run.
- **re-read after compaction**: a true re-read where every overlapping earlier read happened before the latest
  compaction, so the model had lost those lines from its context.
- **budget groups**: solved; unsolved after using (almost) the whole budget (a cap hit, or ≤ 2 of 28 calls left);
  unsolved after stopping early.
- **time shares**: model time is the sum of request latencies over the summed task durations. The model time is then
  split into reasoning and the rest by generated characters.
- **served direction**: the mean log-probability of the model's own held-out output tokens, adapter minus base, scored
  on the server that is actually used. The 95% CI comes from resampling whole sequences.

## What is not here, and why

The competition's tasks, repositories, traces and patches are licensed to entrants only and are not released. Any
number derived from them appears only as an aggregate or against an anonymous task index. There are no credentials,
machine details or logs from the rented machines.

## Data licence

`data/audit.json` refers to SWE-rebench tasks (CC-BY-4.0; Badertdinov et al., 2025, arXiv:2505.20411,
https://huggingface.co/datasets/nebius/SWE-rebench). The adapter was trained on Gemma 4 outputs and is subject to the
Gemma Terms of Use (https://ai.google.dev/gemma/terms) and the Gemma Prohibited Use Policy. The 124 training runs
themselves were not retained, so they are not part of this release. `adapter/c8_manifest.json` records the training
setup: rank 8, alpha 16, q_proj + o_proj, 2 epochs, 44 steps, lr 1e-4, a 4-bit nf4 load of the unquantised QAT
checkpoint, 176 training sequences (2.05 M tokens) and 12 held-out sequences from 8 held-out tasks.
