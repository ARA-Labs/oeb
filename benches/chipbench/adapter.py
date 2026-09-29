"""L0 adapter: Chip-Bench hill-climb run_dir -> canonical record.

Thin adapter over benches/base_adapter.py (as benches/speedrun does): the
run_dir written by benches/chipbench/convert.py is canonical-steps-shaped
(src/msg/tools/obs/session/api_turn/patches), so the base adapter runs
unchanged (scope by tool kind, gid assignment, manifest skeleton, usage
sidecar); then the chipbench rules are applied to the finished unit in
place:

1. WORLD DECLARATION: the agent drives its own shell + edit channel
   against a workspace it controls; authored material is recorded in the
   patch channel (claude-code Edit/Write, grok search_replace/write);
   codex heredoc writes are shell redirections, which the assembler does
   not read. feedback_polarity follows the task's
   metric direction (task.json metric.direction: min -> lower_better,
   max -> higher_better) — the number the agent tracks across `make
   score` reads; the score line next to it is always higher-better and
   is capped in-run (see domain note). in_run_authority task_given=True
   (frozen public harness + `make score` = graded material reachable
   in-run), external=False (no verdict arrives mid-run; hidden grading
   is post-hoc). The hidden grade is written to world.json "outcome" for
   analysis; it never enters the record, and no score or judge reads it.

2. SUBMISSION STEP worded for this world (wall-clock cap, hidden grading
   afterwards).

3. MANIFEST: bench, harness, chip-specific scope rules (EDA tools are
   executions; `taskoutput` polls are readlog; edits are workspace),
   grade block, session stitching.

Scope is the KIND of tool the harness ran (tool name -> TOOL_KIND in the
base adapter): every shell call is env whatever its text says.

Usage: python benches/chipbench/adapter.py <run_dir> <out_dir>
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "base_adapter.py"

POLARITY = {"min": "lower_better", "max": "higher_better"}


def main() -> None:
    run_dir, out_dir = sys.argv[1], sys.argv[2]
    sys.argv = [str(BASE), run_dir, out_dir]
    spec = importlib.util.spec_from_file_location("base_adapter", BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

    run, out = Path(run_dir), Path(out_dir)
    meta = json.loads((run / "meta.json").read_text())
    rec = out / "record.jsonl"
    steps = [json.loads(x) for x in rec.read_text().splitlines() if x.strip()]

    # 2. submission step wording
    sub = steps[-1]
    if sub.get("scope") == "submission":
        sub["action"] = ("wall-clock cap reached; the submission surface on "
                         "disk is graded" if meta.get("hit_budget") else
                         "agent finishes; the submission surface on disk is "
                         "graded")
        sub["obs"] = ("episode ends; hidden grading (hidden workloads/seeds/"
                      "variants) runs afterwards and is not part of this "
                      "record — its verdict is declared in world.json")
    rec.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in steps))

    n_poll = sum(1 for s in steps
                 if (s.get("action_struct") or {}).get("prog") == "(poll)")
    n_patch = sum(1 for s in steps if s.get("patches"))

    # 3. manifest
    mf = out / "record_manifest.json"
    man = json.loads(mf.read_text())
    man["bench"] = "chipbench-hillclimb"
    man["world"] = meta.get("task_name")
    man["task_title"] = meta.get("task_title")
    man["category"] = meta.get("category")
    man["tier"] = meta.get("tier")
    man["scaffold"] = meta.get("agent")
    man["agent_label"] = meta.get("agent_label")
    man["reward_note"] = meta.get("reward_note")
    man["grade"] = meta.get("grade")
    man["metric_direction"] = meta.get("metric_direction")
    man["harness"] = ("Chip-Bench hill-climb driver: isolated workspace "
                      "copy, wall-clock cap, hidden/ vaulted during the run, "
                      "post-hoc hidden grading")
    man["scope_rulings"] = {
        "EDA / simulation / build execution (make score, make check, "
        "verilator, iverilog, yosys, nextpnr, abc, sby, g++, riscv gcc, "
        "python models and search scripts, ad-hoc binaries)":
            "env exec — the design's measured behaviour on the public "
            "workloads is the bench world; only running something "
            "observes it; ALSO when the program sits behind a compound "
            "head — `(time make score)`, `rm ...; for ...; do python3 "
            "...`, `mk(){ yosys ...; }`",
        "background-task polls (claude-code TaskOutput, grok "
        "get_command_or_subagent_output -> `taskoutput {...}`)":
            "env readlog, prog (poll) — watching a launched simulation "
            "and reading what it printed",
        "retrieval and edits (cat/ls/grep reads; Edit/Write/search_replace/"
        "write patches; heredoc writes)":
            "workspace — static source is collected data; an edit settles "
            "nothing until something is run",
        "scaffold bookkeeping (todo lists, ToolSearch, TaskStop/kill)":
            "workspace (rendered under the `todo` head)",
        "the agent's last message": "submission",
        "hidden grading (results/<task>/<agent>.json score, gates, metric)":
            "EXCLUDED from the record — outcome oracle; declared in "
            "world.json outcome",
        "leading `cd <workspace> &&` on grok/codex commands":
            "stripped in rendering (cwd IS the workspace; the prefix hid "
            "the program from the classifier)",
    }
    man["channel_policy"] = ("text+tools (thinking excluded from every "
                             "dialect by policy: grok logs thought text, "
                             "claude logs only token counts, codex nothing)")
    man["channels"] = {
        "official_reply": "absent (single prompt; nothing replies to the "
                          "agent during the run)",
        "transcript_text": "visible",
        "transcript_thinking": "excluded by policy (cross-scaffold symmetry)",
    }
    man["scaffold_caveat"] = (
        "codex batches shell commands per exec step; claude-code and grok "
        "issue one tool call per step, grok often several per API turn — "
        "step counts are a scaffold property, not an agent property")
    man["sessions"] = {
        "transcript_files": meta.get("transcript_files"),
        "anchor": meta.get("transcript_anchor"),
        "resumed": meta.get("resumed"),
        "rate_limit_waits": meta.get("rate_limit_waits"),
        "rule": "all files sharing the anchor's session_id, timestamp "
                "order; each file = one `run` index (session boundary)",
    }
    man["duration_seconds"] = meta.get("duration_seconds")
    man["hit_budget"] = meta.get("hit_budget")
    man["tokens_reported_by_driver"] = meta.get("tokens_reported")
    man["task_prompt"] = meta.get("task_prompt")
    man["task_prompt_source"] = meta.get("task_prompt_source")
    man["task_readme_source"] = meta.get("task_readme_source")
    man["workspace_dir"] = meta.get("workspace_dir")
    man["source_record"] = meta.get("source_record")
    man["source_repo"] = meta.get("source_repo")
    man["n_poll_steps"] = n_poll
    man["n_patch_steps"] = n_patch
    man["climbing_curve"] = ("score_log.jsonl (the agent's own `make score` "
                             "lines, copied from its workspace) — ground "
                             "truth for checking claimed metric moves"
                             if (run / "score_log.jsonl").exists() else
                             "absent (workspace grade/score_log.jsonl not "
                             "committed)")
    man["textualizer"] = ("benches/base_adapter.py, then "
                          "benches/chipbench/adapter.py: scope by tool kind "
                          "only, world redeclaration, submission wording, "
                          "grade/session blocks")
    mf.write_text(json.dumps(man, indent=1, ensure_ascii=False))

    # 1. world declaration
    g = meta.get("grade") or {}
    direction = meta.get("metric_direction")
    world = {
        "scope_default": "agent",
        "authored_channel": {"kind": "patches+shell_redirect"},
        "action_genre": "dev_shell",
        "task_apparatus": (
            "frozen public harness in the workspace (checksummed; `make "
            "score` refuses to run if edited): simulators / synthesis / "
            "formal tools measure the design on the PUBLIC workloads and "
            "append one line per run to build/score_log.jsonl; the hidden "
            "grader re-measures on hidden variants after the run"),
        "in_run_authority": {"task_given": True, "external": False},
        "outcome": {
            "status": "graded",
            "score": g.get("score"),
            "score_rule": g.get("score_rule"),
            "metric_name": g.get("metric_name"),
            "metric_value": g.get("metric_value"),
            "metric_direction": direction,
            "baseline": g.get("baseline"),
            "reference": g.get("reference"),
            "failed_gates": g.get("failed_gates"),
            "audit_flags": g.get("audit_flags"),
            "note": "the benchmark's hidden-grader verdict, recorded for "
                    "analysis only: the agent never saw it, and no score "
                    "or judge reads it. 1.0 = matches the best verified "
                    "design known for the task",
        },
        "declared_by": "benches/chipbench/adapter.py",
    }
    if not g.get("score_rule"):
        # a set-aside second run's record carries no score_rule: it is
        # graded under the benchmark's earlier rule only
        world["outcome"]["score_rule"] = (
            "the benchmark's earlier rule (clamped at 1.25, original expert "
            "references); this record was never regraded")
        world["outcome"]["note"] += ("; this score is not comparable with "
                                     "scores under the current rule (the raw "
                                     "metric_value is)")
    if direction in POLARITY:
        world["feedback_polarity"] = POLARITY[direction]
    (out / "world.json").write_text(json.dumps(world, indent=1,
                                               ensure_ascii=False))

    for name in ("domain_note.txt", "score_log.jsonl"):
        if (run / name).exists():
            shutil.copyfile(run / name, out / name)
    print(f"chipbench rules applied: world declared "
          f"({world.get('feedback_polarity', 'polarity omitted')}, "
          f"score {g.get('score')}), {n_poll} polls, {n_patch} patch steps")


if __name__ == "__main__":
    main()
