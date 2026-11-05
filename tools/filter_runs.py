"""Self-distillation step 3: decide which solved runs become training examples.

A run is kept only if ALL hold:
  1. resolved by the official verifier, and its task passed the gold check (bare fails, gold passes);
  2. no repeated-call loop (the same tool call with the same arguments 3 times in a row);
  3. test-file rule (a strict test-file pattern): editing an EXISTING test/config file rejects the
     run; creating a NEW test file (a reproduction test) is allowed;
  4. read-before-edit: every `edit_file` on an existing file targets a file the agent read or named in a command earlier;
  5. no look into other history (`git log/show/checkout` with --all / --tags / reflog / other refs) — the images are
     scrubbed, this is a second line of defence;
  6. (v3, after an audit of 20 runs by a separate LLM agent: ~35-40% of v2-kept runs were poor examples) habit and record checks: the tool budget was not exhausted; a python/pytest command ran
     after the last source edit; no malformed call (argument name outside the tool schema, tool-call syntax leaked into
     an argument or the text, a call the harness refused for missing parameters); no two identical calls in a row
     (submit_patch / get_status excepted); the run ends with a text answer and called submit_patch.
     `--v2` turns these off. Scratch files left in the final patch are reported, not dropped.
"Existing" comes from the repo's file list recorded at task start (`files/<id>.txt`, `git ls-files` in the wrapper);
for runs made before that was recorded, from the final patch ("new file mode" = created by the agent).

    python tools/filter_runs.py RUNS_DIR --out kept.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
import zlib
from pathlib import Path

# Rule: no edits to existing tests, extended with conftest.py and test-runner config files: the harness strips those from the
# scored patch anyway, so editing them teaches a habit that never counts.
TEST_PATH_REGEX = re.compile(r'(^|/)(tests?/.*|test_[^/]+\.py|[^/]+_test\.py|conftest\.py|pytest\.ini|tox\.ini)$', re.IGNORECASE)
GIT_OTHER_HISTORY = re.compile(r"git\s+(log|show|checkout|switch|diff|reflog|tag|branch)\b[^|;&]*(--all|--tags|--branches|reflog|origin/|@\{)")


def norm(p: str) -> str:
    p = (p or "").strip()
    for pre in ("/workspace/", "/testbed/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def harness_protected(rel_path: str) -> bool:
    """Files the scorer strips from a patch (test files and test-runner configuration), mirrored from the harness."""
    p = Path(rel_path)
    name = p.name
    if name in {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg", ".pytest.ini", "sitecustomize.py",
                "usercustomize.py"} or name.endswith(".pth"):
        return True
    if name.startswith("_") and name.endswith("_stubs.py"):     # stub modules the harness writes into the sandbox
        return True
    if (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py"):
        return True
    return bool({"tests", "test", "testing"} & {x.lower() for x in p.parts[:-1]} and name.endswith(".py"))


def is_protected(path: str) -> bool:
    return bool(TEST_PATH_REGEX.search(path)) or harness_protected(path)


# a shell command that may change files (edits through run_command would otherwise bypass the rule)
WRITE_CMD = re.compile(r"sed\s+-i|>>?|\btee\b|\brm\b|\bmv\b|\bcp\b|\btruncate\b|\bpatch\b|git\s+(mv|rm|checkout|restore|apply|stash)|"
                       r"open\([^)]*['\"][wa]")


def is_valid_edit_target(target_path: str, initial_repo_files: set[str]) -> bool:
    """Reject edits to existing test files; allow new test files and source edits.
    'Test file' = the strict test-file pattern OR the harness's own protected test/config rule."""
    is_test_file = is_protected(target_path)
    file_existed_at_start = target_path in initial_repo_files
    if is_test_file:
        return not file_existed_at_start
    return True


CHECK_CMD = re.compile(r"\b(python3?|pytest|py\.test|tox|nox|make\s+test)\b")
LEAK = re.compile(r'<\|"\|>|`,\s*\w+:|<\|tool_call|<\|channel|<tool_call\|>')
TEXT_LEAK = re.compile(r"<\|tool_call|<\|channel|<channel\|>|\bcall:\w+\{")
SCRATCH = re.compile(r"(^|/)(repro|reproduce|debug|scratch|tmp|temp)[^/]*\.py$", re.IGNORECASE)
# Harness-specific strings (env vars; see harness_string): the refusal text for a call with missing parameters, the
# marker of an exhausted tool budget, and the first words of the summarisation prompt.


def harness_string(var: str, what: str) -> str:
    """Harness-specific text is not shipped with this release: set it from your copy of the competition harness."""
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"set {var} to {what} (from the competition harness; not included in this release)")
    return v


def quality(bodies: list[dict], patch: str) -> tuple[list[str], list[str]]:
    """v3 habit/record checks -> (drop reasons, report-only notes)."""
    reasons, notes = [], []
    calls = sorted(bodies, key=lambda d: d.get("seq") or 0)
    schema = {}
    for d in calls:
        for t in d.get("tools") or []:
            f = t.get("function") or {}
            schema[f.get("name")] = set((f.get("parameters") or {}).get("properties", {}))
    results = {}                                            # tool_call_id -> result text (from later requests)
    for d in calls:
        for m in d.get("messages") or []:
            if m.get("role") == "tool" and m.get("tool_call_id"):
                results[m["tool_call_id"]] = str(m.get("content"))
    seq, final_text = [], None                              # (name, args, result) in order, from the responses
    for d in calls:
        msg = ((d.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
        ms = d.get("messages") or []
        if len(ms) == 1 and str(ms[0].get("content", "")).startswith(SUMMARY):
            continue                                        # compaction summary call
        for txt in (msg.get("content"), msg.get("reasoning_content")):
            if txt and TEXT_LEAK.search(txt):
                reasons.append("malformed: tool-call syntax in the text")
        if not msg.get("tool_calls"):
            final_text = msg.get("content") or final_text
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name, raw = fn.get("name") or "", fn.get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except Exception:
                args = {"_raw": raw}
            args = args if isinstance(args, dict) else {"_raw": args}
            res = results.get(tc.get("id"), "")
            if name in schema and set(args) - schema[name]:
                reasons.append(f"malformed: argument {sorted(set(args) - schema[name])[0]!r} not in {name}'s schema")
            if any(isinstance(v, str) and LEAK.search(v) for k, v in args.items() if k not in ("content", "new_string", "old_string")) \
                    or any(isinstance(args.get(k), str) and re.search(r'<\|"\|>|<\|tool_call', args[k]) for k in ("content", "new_string", "old_string")):
                reasons.append("malformed: tool-call syntax inside an argument")
            if REFUSED in res:
                reasons.append("malformed: call refused for missing parameters")
            if BUDGET in res:
                reasons.append("tool budget exhausted")
            seq.append((name, json.dumps(args, sort_keys=True), res))
    for a, b in zip(seq, seq[1:]):
        if a[:2] == b[:2] and a[0] not in ("submit_patch", "get_status"):
            reasons.append("two identical calls in a row"); break
    patched = set(re.findall(r"(?m)^diff --git a/\S+ b/(\S+)", patch or ""))
    last_edit = max((i for i, (n, a, _) in enumerate(seq) if n in ("edit_file", "write_file")
                     and norm(json.loads(a).get("filepath", "")) in patched), default=None)
    if last_edit is not None and not any(n == "run_command" and CHECK_CMD.search(json.loads(a).get("command", ""))
                                         for n, a, _ in seq[last_edit + 1:]):
        reasons.append("no python/pytest run after the last source edit")
    if not any(n == "submit_patch" for n, _, _ in seq):
        reasons.append("never called submit_patch")
    if not final_text:
        reasons.append("no final text answer")
    created = re.findall(r"(?m)^diff --git a/\S+ b/(\S+)\n(?:new file mode)", patch or "")
    notes += [f"scratch file left in the patch: {f}" for f in created if SCRATCH.search(f)]
    return sorted(set(reasons)), notes


def tool_calls(bodies: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for d in sorted(bodies, key=lambda d: d.get("seq") or 0):
        msg = ((d.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"_raw": fn.get("arguments")}
            out.append((fn.get("name") or "", args if isinstance(args, dict) else {"_raw": args}))
    return out


def initial_files(wdir: Path, task: str, patch: str) -> tuple[set[str], str]:
    f = wdir / "files" / f"{task}.txt"
    if f.exists() and f.read_text().strip():          # an empty list (git ls-files failed) counts as missing
        return {l.strip() for l in f.read_text().splitlines() if l.strip()}, "recorded"
    # fallback: everything the final patch touches is "existing" unless the patch creates it
    created, touched = set(), set()
    for block in re.split(r"(?m)^diff --git ", patch or ""):
        m = re.match(r"a/(\S+) b/(\S+)", block)
        if not m:
            continue
        touched.add(m.group(2))
        if re.search(r"(?m)^new file mode", block):
            created.add(m.group(2))
    return touched - created, "from-final-patch"


def judge(task: str, row: dict, bodies: list[dict], wdir: Path, v3: bool = True) -> dict:
    reasons = []
    if not row.get("resolved"):
        reasons.append("not resolved")
    if row.get("gold_ok") is False:
        reasons.append("gold check failed")
    tcs = tool_calls(bodies)
    if not tcs:
        reasons.append("no logged tool calls")
    if any(tcs[i] == tcs[i + 1] == tcs[i + 2] for i in range(len(tcs) - 2)):
        reasons.append("repeated-call loop")
    patch = (wdir / f"patch_{task}.diff").read_text() if (wdir / f"patch_{task}.diff").exists() else ""
    init, init_src = initial_files(wdir, task, patch)
    seen: set[str] = set()
    for name, args in tcs:
        if name == "read_file":
            seen.add(norm(args.get("filepath") or args.get("path") or ""))
        if name == "run_command":
            cmd = args.get("command", "")
            if GIT_OTHER_HISTORY.search(cmd):
                reasons.append("looked at other git history")
            if WRITE_CMD.search(cmd):
                for tok in re.findall(r"[\w./-]+", cmd):
                    t = norm(tok)
                    if t in init and is_protected(t):
                        reasons.append(f"command may change existing test/config file {t}")
            seen.update(norm(m) for m in re.findall(r"[\w./-]+\.\w+", cmd))
        if name in ("edit_file", "write_file"):
            p = norm(args.get("filepath") or args.get("path") or "")
            if not is_valid_edit_target(p, init):
                reasons.append(f"edited existing test/config file {p}")
            if name == "edit_file" and p in init and not any(p.endswith(s) or s.endswith(p) for s in seen if s):
                reasons.append(f"edited {p} before reading it")
    notes = []
    if v3:
        q, notes = quality(bodies, patch)
        reasons += q
    return {"task": task, "keep": not reasons, "reasons": sorted(set(reasons)), "notes": notes, "tool_calls": len(tcs),
            "initial_files_source": init_src}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", help="folder with w*/ worker folders (runs.jsonl, bodies.jsonl.gz, patch_*.diff)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manual-drops", help="JSON {task: reason} of runs dropped after a check the filter cannot make")
    ap.add_argument("--v2", action="store_true", help="v2 rules only (no habit/record checks)")
    a = ap.parse_args(argv)
    global REFUSED, BUDGET, SUMMARY
    REFUSED = harness_string("HARNESS_REFUSED_TEXT", "the harness's refusal text for a call with missing parameters")
    BUDGET = harness_string("HARNESS_BUDGET_MARKER", "the marker the harness puts in a result when the tool budget is exhausted")
    SUMMARY = harness_string("HARNESS_SUMMARY_PREFIX", "the first words of the harness's summarisation prompt")
    drops = json.loads(Path(a.manual_drops).read_text()) if a.manual_drops else {}
    results = []
    for wdir in sorted(Path(a.run_root).glob("w*")):
        rows = {json.loads(l)["task"]: json.loads(l) for l in open(wdir / "runs.jsonl")} if (wdir / "runs.jsonl").exists() else {}
        bodies: dict[str, list] = {}
        try:
            with gzip.open(wdir / "bodies.jsonl.gz", "rt") as g:
                for l in g:
                    d = json.loads(l)
                    bodies.setdefault(d.get("task"), []).append(d)
        except (EOFError, OSError, json.JSONDecodeError, zlib.error):
            pass      # a worker stopped mid-write: every complete line before the cut is kept (zlib.error)
        for t, lst in bodies.items():                # a resumed task logs a second attempt with seq restarting: keep the last
            starts = [i for i, d in enumerate(lst) if i == 0 or (d.get("seq") or 0) <= (lst[i - 1].get("seq") or 0)]
            bodies[t] = lst[starts[-1]:]
        for task, row in rows.items():
            if row.get("prepare_stage") == "ready" and row.get("resolved"):
                r = judge(task, row, bodies.get(task, []), wdir, v3=not a.v2) | {"worker": wdir.name}
                if task in drops:
                    r["keep"] = False
                    r["reasons"] = sorted(set(r["reasons"]) | {f"manual: {drops[task]}"})
                results.append(r)
    kept = [r for r in results if r["keep"]]
    Path(a.out).write_text(json.dumps({"solved": len(results), "kept": len(kept), "runs": results}, indent=1))
    print(f"solved {len(results)}, kept {len(kept)}")
    for r in results:
        print(f"  {'KEEP' if r['keep'] else 'drop'} {r['task']}: {', '.join(r['reasons']) or 'ok'} ({r['initial_files_source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
