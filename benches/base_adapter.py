"""L0 base adapter: a converted run_dir -> canonical scope-tagged record.

The shared adapter the benchmark adapters build on. A run is one agent
working one task inside a stateful container or workspace through a CLI
agent harness; any grading that happens after the run is not part of the
record. Input is a run_dir of canonical steps (steps.json + meta.json, with
full observations) written by a benchmark's converter
(benches/speedrun/convert.py, benches/chipbench/convert.py). The
benchmark adapters (benches/speedrun/adapter.py,
benches/chipbench/adapter.py) run this one and then apply their own rules;
benches/posttrainbench/adapter.py imports its tool-kind table, and it and
benches/speedrun/convert.py share its replay rule and chars-per-token band.

Scope rules (recorded in the manifest): the container's RUN behavior is the
bench world; only execution observes it.
  interpreter/tool execution (python, pytest, pip, node, make,
  task CLIs, ad-hoc scripts, package installs, process polls)   -> env
  retrieval + edits (ls/cat/grep/find/sed -n/git reads, file
  writes, patches, todo bookkeeping)                            -> workspace
  the agent's last message                                      -> submission

Scope is the KIND of tool the harness ran, a fact of the call's NAME
(TOOL_KIND), never a reading of the command's text. A shell
call and a poll of a running process observe the container (env); a read,
an edit, a delegation and bookkeeping do not (workspace). A tool name the
table does not list stops the adapter (UnknownTool) and is added to the
table by hand.

Cross-scaffold symmetry: scaffolds differ (codex batches many shell commands
per exec step; claude-code issues one tool call per step), so step COUNTS are
a scaffold property, never an agent property. Thinking is excluded for every
scaffold by policy (not every scaffold logs it).

Usage: python benches/base_adapter.py <run_dir> <out_dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Chars per output token, used where a dump reports only run-total output
# tokens and the visible channel has to be converted to tokens: 2.9 is
# calibrated on runs with exact per-call token counts, 4.2 is the generic
# English-prose ratio. Converters report both ends of this band.
CPT_BAND = (2.9, 4.2)
CPT = CPT_BAND[0]

# Transcript replays: a harness that resumes a session may re-emit its
# earlier history as if new. A block of at least REPLAY_MIN consecutive
# steps identical to an earlier contiguous block is such a replay.
REPLAY_MIN = 8


def replay_steps(keys: list, can_start) -> set:
    """Indices of replayed steps. keys[i] is step i's identity (whatever
    the converter compares); a replayed block may begin at i only when
    can_start(i). A stalled poll loop (the same few steps repeating) is
    not a replay: a block needs at least 4 distinct steps."""
    first: dict = {}
    drop: set = set()
    i = 0
    while i < len(keys):
        j = first.get(keys[i])
        if j is not None and can_start(i):
            n = 0
            while i + n < len(keys) and j + n < i and keys[i + n] == keys[j + n]:
                n += 1
            if n >= REPLAY_MIN and len(set(keys[i:i + n])) >= 4:
                drop.update(range(i, i + n))
                i += n
                continue
        first.setdefault(keys[i], i)
        i += 1
    return drop


def _txt(v):
    """A field as text, whole. The record keeps full text; readers that
    need less (the judges' prompts) clip at read time, so no channel is
    cut in favor of another."""
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False,
                                                default=str)
    return s


class UnknownTool(KeyError):
    """A tool name TOOL_KIND does not list. Loud on purpose: the scope of
    a step is a fact of the tool's name, and a name nobody has looked at
    gets no scope by guesswork — it gets added to the table."""


# Whether a step observed the container is a fact the harness knows:
# which kind of tool it ran. A shell call and a poll of a running process
# observe the container (env); a file read, an edit, a delegation and
# bookkeeping do not (workspace). The tool's NAME decides, never the text
# of the command. What a shell call showed or launched (a `cat` of a
# training log, a `python` run) is the extractor's reading of its action
# and observation. The names cover the harnesses the three benchmarks'
# runs use: claude-code / kimi-code, codex, grok-cli, pi, qwen-code,
# prime-agent, opencode (todowrite, task, codesearch) and cursor-cli
# (<name>ToolCall).
TOOL_KIND = {
    "shell": {"Bash", "bash", "exec", "exec_command", "run_shell_command",
              "run_terminal_command",
              "shellToolCall",
              "ipython"},            # prime-agent's one tool: a code cell
    "poll": {"wait", "wait_agent", "write_stdin", "TaskOutput", "BashOutput",
             "get_command_or_subagent_output", "Monitor", "monitor",
             "awaitToolCall"},
    "read": {"Read", "read", "read_file", "Grep", "grep",
             "Glob", "glob", "list_dir", "list_directory",
             "search_file_content", "ReadMediaFile", "ListAgents",
             "list_agents", "WebFetch", "WebSearch", "websearch",
             "web_fetch", "FetchURL", "codesearch",
             "readToolCall", "globToolCall", "grepToolCall",
             "webSearchToolCall", "webFetchToolCall", "webfetch"},
    "edit": {"Edit", "MultiEdit", "edit", "Write", "write", "write_file",
             "search_replace", "apply_patch",
             "file_change", "NotebookEdit", "editToolCall"},
    "delegate": {"Agent", "spawn_agent", "spawn_subagent", "AgentSwarm",
                 "send_message", "SendMessage", "followup_task",
                 "Task", "task"},
    "bookkeeping": {"TodoWrite", "TodoList", "todowrite", "todoread", "TaskCreate",
                    "TaskUpdate", "TaskList", "TaskGet", "TaskStop",
                    "task_stop", "todo_write", "get_goal", "update_goal",
                    "create_goal", "record_artifact", "manage_task",
                    "updateTodosToolCall",
                    "schedule", "ScheduleWakeup", "ToolSearch", "KillShell",
                    "kill_command_or_subagent", "interrupt_agent",
                    "CronList"},
}
KIND_OF = {n: k for k, names in TOOL_KIND.items() for n in names}
KIND_RANK = ["shell", "poll", "delegate", "edit", "read", "bookkeeping"]


def _call_json(cmd: str) -> dict:
    """The JSON argument of a call rendered `<name> {...}`; {} when the
    call carries none (or the dump cut it)."""
    i, j = cmd.find("{"), cmd.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        v = json.loads(cmd[i:j + 1])
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def tool_kind(fn, cmd: str) -> str:
    """The kind of tool the harness ran. write_stdin sends input to a
    live process: with input it drives the process (shell), with none
    it only reads what the process printed since (poll) — a field of
    the call, read from the call's JSON."""
    k = KIND_OF.get(str(fn or ""))
    if k is None:
        raise UnknownTool(f"{fn!r} (call: {str(cmd)[:120]!r}) is not in "
                          "TOOL_KIND — add the name to its kind")
    if str(fn) == "write_stdin" and _call_json(cmd or "").get("chars"):
        return "shell"
    return k


def _exec_struct(cmd: str, kind: str) -> dict:
    """The harness facts of an env step: its kind (exec | readlog), the
    program head and the files the command names; a poll names the
    session it polls when its JSON says so."""
    if kind == "readlog":
        st = {"kind": kind, "prog": "(poll)", "target": ""}
        j = _call_json(cmd)
        for k in ("session_id", "cell_id", "task_id"):
            if j.get(k) not in (None, ""):
                st["target"] = f"session {j[k]}"
                break
        return st
    head = cmd.strip().split()[0] if cmd.strip() else "(exec)"
    targets = re.findall(
        r"[\w./-]+\.(?:py|sh|json|csv|ya?ml|toml|md|ipynb|png|jpg|pdf|txt)\b",
        cmd)[:4]
    return {"kind": kind, "prog": head[:40],
            "target": " ".join(targets)[:160]}


def build_steps(run: Path):
    raw = json.loads((run / "steps.json").read_text())
    meta = json.loads((run / "meta.json").read_text())

    steps = []
    turn = 1
    pending_delib = []

    def push(scope, action, reasoning, obs, struct=None, sess=0,
             aturn=None):
        # "run" = the session index the converter stamped, so readers
        # see where a session restarted
        nonlocal turn
        st = {"turn": turn, "run": sess, "level": 1, "scope": scope,
              "action": action, "reasoning": reasoning,
              "testing": None, "expected": None, "surprised": False,
              "rejected": None, "obs": obs}
        if aturn is not None:
            # api_turn = converter's API-call stamp: joins record gids
            # to usage.json turns for the T1 token-budget reading.
            # Distinct from this adapter's own env-step "turn" counter.
            st["api_turn"] = aturn
        if struct is not None:
            # the struct is a harness fact (kind / program / files
            # named); the record's observation channel carries ONLY
            # what the world returned
            st["action_struct"] = struct
        steps.append(st)
        if scope == "env":
            turn += 1
        return st

    last_msg = ""
    by_fact = [0]
    pending_actor = None
    pending_sess = 0
    pending_turn = None
    for s in raw:
        if s.get("src") in ("user", "system"):
            # a message the agent RECEIVED: its own step, the text in
            # the observation channel, kept apart from the agent's words
            # (it carries compaction summaries and workers' completion
            # notices). It does not consume the agent's pending
            # deliberation: that still rides the agent's own next action.
            st = push("workspace", "(harness message)",
                      str(s.get("note") or ""), _txt(s.get("msg")),
                      sess=s.get("session") or 0, aturn=s.get("api_turn"))
            st["tool_kind"] = "message"
            st["actor"] = s.get("actor") or "harness"
            if s.get("delegation"):
                st["delegation"] = s["delegation"]
            continue
        msg = str(s.get("msg") or "")
        if msg:
            last_msg = msg
        tools = s.get("tools") or []
        actor = s.get("actor")

        # a message is deliberation carried into the SAME actor's next
        # action; across an actor boundary (a worker's closing report
        # before the parent's next call, or the parent's text before an
        # inlined worker step) it is its own step, never folded into
        # another actor's reasoning
        if pending_delib and pending_actor != actor:
            st = push("workspace", "(message; no action)",
                      "\n".join(pending_delib), "",
                      sess=pending_sess, aturn=pending_turn)
            st["tool_kind"] = "message"
            if pending_actor:
                st["actor"] = pending_actor
            pending_delib = []
        if not tools:
            if msg:
                pending_delib.append(msg)
                pending_actor = actor
                pending_sess = s.get("session") or 0
                pending_turn = s.get("api_turn")
            continue

        # the step's obs is the combined output of its tool calls; it lands
        # on the last one, the earlier calls carry the reasoning forward
        cmds = [str(t.get("cmd") or "") for t in tools]
        # the step's kind is its strongest call's (a shell call beside a
        # read is a shell step)
        kinds = [tool_kind(t.get("fn"), c) for t, c in zip(tools, cmds)]
        by_fact[0] += len(kinds)
        step_kind = min(kinds, key=KIND_RANK.index)
        step_scope = "env" if step_kind in ("shell", "poll") else "workspace"
        obs = _txt(s.get("obs")) if s.get("obs") is not None else ""
        reasoning = "\n".join(pending_delib + ([msg] if msg else [])) \
            if (pending_delib or msg) else ""
        pending_delib = []

        shown = " ; ".join(_txt(c) or "(poll)" for c in cmds)
        env_i = next((i for i, k in enumerate(kinds) if k in ("shell", "poll")), None)
        st = push(step_scope, f"$ {shown}", reasoning, obs,
                  _exec_struct(cmds[env_i], "readlog" if kinds[env_i] == "poll" else "exec")
                  if env_i is not None else None,
                  sess=s.get("session") or 0,
                  aturn=s.get("api_turn"))
        st["tool_kind"] = step_kind
        if s.get("patches"):        # edit content: the extractor
            st["patches"] = s["patches"]   # reads it, the assembler joins
                                           # edited files to acts
        for k in ("actor", "delegation", "payload"):   # converter-level truths:
            if s.get(k):                   # WHO acted (a worker), which steps
                st[k] = s[k]               # dispatch/relay a worker, and whether
                                           # a call that writes carried its body

    meta["scope_by"] = {"tool_kind": by_fact[0]}
    return steps, last_msg, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    run, out = Path(a.run_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps, last_msg, meta = build_steps(run)

    # the closing step carries the agent's last message and the one
    # harness fact that the transcript ended. What happens after it
    # (a budget cut, graders, verifiers) is the bench's world
    # declaration (world.json / manifest), not something the agent
    # observed.
    steps.append({"turn": max((s["turn"] for s in steps), default=0),
                  "run": max((s["run"] for s in steps), default=0),
                  "level": 1, "scope": "submission",
                  "action": "(transcript ends)",
                  "reasoning": last_msg,
                  "testing": None, "expected": None, "surprised": False,
                  "rejected": None, "tool_kind": "message",
                  "obs": ""})
    for i, s in enumerate(steps):
        s["gid"] = i

    n_env = sum(1 for s in steps if s["scope"] == "env")
    n_ws = sum(1 for s in steps if s["scope"] == "workspace")
    manifest = {
        "bench": "base",
        "world": meta.get("task_name"),
        "model": meta.get("model"),
        "scaffold": meta.get("agent"),
        "trial": meta.get("trial_name"),
        "reward": meta.get("reward"),
        "duration_seconds": meta.get("duration_seconds"),
        "hit_budget": meta.get("hit_budget"),
        "n_env_steps": n_env,
        "scope_by": meta.get("scope_by"),
        "scope_law": ("scope is the KIND of tool the harness ran (tool_kind on "
                      "every step): shell calls and polls of a running process "
                      "observe the container (env); reads, edits, delegations "
                      "and bookkeeping do not (workspace). Never a reading of "
                      "the command text — there is no text fallback; a tool "
                      "name the table does not know stops "
                      "the adapter. What a step showed or launched is the "
                      "extractor's reading of its action and observation"),
        "n_workspace_steps": n_ws,
        "scope_rulings": {
            "interpreter/tool execution (python, pytest, pip, node, make, "
            "task CLIs, ad-hoc scripts, installs, process polls)":
                "env — the container's RUN behavior is the bench world; "
                "only execution observes it",
            "retrieval and edits (ls/cat/grep/find/sed -n/git reads, file "
            "writes, patches, todo bookkeeping)":
                "workspace — static source/data is collected data; editing "
                "settles nothing behavioral until something is run",
            "the agent's last message": "submission",
            "grading after the run (never shown to the agent)":
                "EXCLUDED — the outcome never enters the record",
        },
        "harness": "installed CLI agent in a container",
        "channel_policy": "text+tools (thinking excluded for every scaffold "
                          "by policy: not every scaffold logs it)",
        "channels": {
            "official_reply": "absent (nothing replies to the agent "
                              "during the run)",
            "transcript_text": "visible",
            "transcript_thinking": "excluded by policy (cross-scaffold "
                                   "symmetry)",
        },
        "scaffold_caveat": "codex batches many shell commands per exec step; "
                           "claude-code issues one tool call per step — step "
                           "counts are a scaffold property, not an agent "
                           "property",
        "task_prompt": meta.get("task_prompt"),
        "token_usage": ("usage.json (per-api-turn, converter-extracted "
                        "from the run's own raw stream; record steps "
                        "carry api_turn joins)"
                        if (run / "usage.json").exists() else
                        "absent — this dump reports no usage; "
                        "budget readings withhold, no char proxy"),
        "textualizer": "benches/base_adapter.py (scope by tool kind only, "
                       "no text fallback; payload verdicts ride edit steps)",
    }
    (out / "record.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in steps))
    (out / "record_manifest.json").write_text(json.dumps(manifest, indent=1))
    # token-usage sidecar: rides converter -> unit verbatim; without it
    # the T1 token-budget reading abstains (no proxy)
    uf = run / "usage.json"
    if uf.exists():
        (out / "usage.json").write_text(uf.read_text())
    # WORLD DECLARATION (the bench declares the world's shape, the scorer
    # never guesses it; the benchmark adapters override what differs):
    #   feedback_polarity higher_better — the default direction of a
    #     task's reward (copied into the graph's outcome block).
    #   authored_channel patches+shell_redirect — agent-written material
    #     is recorded in the patch channel (the assembler builds the
    #     graph's ARTIFACT nodes from the patches; redirections are not
    #     read).
    #   scope_default, action_genre, in_run_authority — descriptive
    #     fields for readers outside the scorer; no score reads them.
    (out / "world.json").write_text(json.dumps({
        "scope_default": "agent",
        "feedback_polarity": "higher_better",
        "authored_channel": {"kind": "patches+shell_redirect"},
        "action_genre": "dev_shell",
        "in_run_authority": {"task_given": True, "external": True},
        "declared_by": "benches/base_adapter.py"}, indent=1))
    print(f"{run.name}: {len(steps)} steps ({n_env} env / {n_ws} ws / "
          f"1 submission) -> {out}")


if __name__ == "__main__":
    main()
