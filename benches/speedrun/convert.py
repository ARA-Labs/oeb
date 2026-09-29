"""speedrun convert: sanitized trace dump -> canonical run_dir.

Input is a PrimeIntellect frontier-automated-speedrun sanitized trace
dump; output is a run_dir (steps.json + meta.json + usage.json) for
benches/speedrun/adapter.py.

Source: github.com/PrimeIntellect-ai/frontier-automated-speedrun traces/.
One run = one agent on one 8xH200 node iterating on the nanoGPT optimizer
speedrun (program.md rulebook) unattended for days. The dump files read:
  events-<id>.json.gz    {"run": id, "events": [...]}  main transcript
  subagents-<id>.json.gz {"index": [...], "events": {child_id: [...]}}
  manifest.json.gz       per-run metadata (model, harness, records, hours)

Event schema (both scaffolds): {"type": text|thinking|tool_use|tool_result,
"role", "ts", "turn", "i", ...}. tool_use carries tool + summary (full
command text) and, for claude-code Edit/Write, file/old/new. thinking is
EXCLUDED by policy (the record keeps text and tool calls; several
harnesses log thinking only redacted or summarised, so excluding it
keeps harnesses comparable).

WORKERS (subagents) ARE INLINED: child transcripts are converted with
the same walk and placed in the main sequence at their own time.
Rationale: (1) children are the same model - the run's own work, billed
to the run; (2) harnesses stay comparable - codex monitors its training
runs (wait polls, log reads) inline in its main trace, claude-code
delegates the identical work to children;
(3) first-hand evidence - without the children, the parent's obs channel is
child-WRITTEN reports (same-model prose), not first-hand world output;
inlining keeps the raw outputs (run.sh logs, verify.py output) that
juries can check reports against.

Marking: every worker step says who acted
and on what - "[worker <id> begins; task: ...]" on its first step,
"[worker <id> on task: ...]" on the rest - and carries actor=worker:<id>;
the parent's DISPATCH step (the call that started the worker) and its
RELAY steps (calls whose result carries the worker's report or status)
are marked in the reasoning channel and carry delegation=dispatch|relay|
dispatch+relay. A dispatch or a relay is not a world observation - the
worker's own runs are - so the bench adapter scopes them workspace. See
_inline_children for linking and placement.

Rendering: the run's manifest names the harness, the harness names its
table (DIALECTS), the table names every tool the agent may call and how
each call is rendered — a tool the table does not list stops the
conversion (UnknownTool) instead of being rendered from whatever field
it happens to carry. Every call that
writes carries a PAYLOAD verdict on its step (carried | truncated |
absent) and the run's tally rides meta.dump_defects, next to the count
of fields the release itself cut ("…[+N chars]"), the calls the dump
never answers and the replays dropped: what the dump lacks is a number
in meta, never silence. Shared spelling across harnesses: a raw
shell line; `cat <f>`; `edit <f>` / `write <f>` + patches; `subagent
<task>`; `todo <text>`; codex `wait {...}` raw (-> readlog poll).

Usage:
  python benches/speedrun/convert.py <traces_dir> <run_id> --out <runs_root>
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

# the shared replay rule and chars-per-token band (benches/base_adapter.py)
_spec = importlib.util.spec_from_file_location(
    "base_adapter", Path(__file__).resolve().parent.parent / "base_adapter.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# The converter cuts nothing. codex wait-polls are all kept - each
# carries an evolving training-log tail.

PATCH_FILE_RE = re.compile(
    r"\*\*\* (Add|Update|Delete) File: (\S+)\n(.*?)(?=\*\*\* (?:Add|Update|"
    r"Delete) File:|\*\*\* End Patch|\Z)", re.S)

# The release's own truncation marker: a field the sanitizer cut ends in
# "…[+2809 chars]". It sits in results, in the agent's prose, in call
# summaries and in edit bodies. The converter repairs nothing (there is
# nothing to repair from): the text rides as the dump has it, marker
# included, and every cut is counted by field in meta.dump_defects.
CUT_RE = re.compile(r"…\[\+\d+ chars\]")


def _cut(*vals) -> bool:
    return any(isinstance(v, str) and CUT_RE.search(v) for v in vals)


def _load(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


class UnknownTool(KeyError):
    """A tool name the harness's table does not list. Loud on purpose:
    rendering an unlisted call from whichever field happens to exist
    can lose its payload silently. A new harness, or a new tool on an
    old one, is added to its table by hand, with the field its payload
    rides in named there."""


class UnknownHarness(KeyError):
    pass


# ---------------------------------------------------------------------
# Rendering tables. The manifest names the harness; the
# harness names its table; the table names every tool the agent may
# call and how its call is rendered. Nothing is guessed from which
# fields a call happens to carry. Each renderer returns
#     (cmd, patches, payload)
# cmd      the action as one line, in the shell-ish spelling every
#          harness shares (`cat f`, `edit f`, `write f`, `subagent t`,
#          `todo t`, a raw shell line);
# patches  the action behind a call that writes — [{path, old, new}],
#          [{path, new}], [{path, patch}] or [{path, edits}] — or None;
# payload  the PAYLOAD CONTRACT, for a call that writes:
#            carried    the body is in the record;
#            truncated  the body is there but the release cut it
#                       (CUT_RE inside it);
#            absent     the dump kept no body — the record says THAT
#                       the file changed, not what changed;
#          None for a call that writes nothing.
# The verdict rides the step (`payload`) and the tally rides meta
# (`dump_defects.payloads`), so a body the dump lacks is a number in
# meta, never an empty string read as "no change".
# ---------------------------------------------------------------------

def _s(e: dict) -> str:
    return str(e.get("summary") or "")


def _a(e: dict) -> dict:
    a = e.get("args")
    return a if isinstance(a, dict) else {}


_QUERY_RE = re.compile(r"<user_query>(.*?)</user_query>", re.S)


def _tagged_query(txt: str):
    """The user's query when the harness tags it (grok-cli: <user_query>),
    verbatim — the task as the agent read it. grok-cli delivers the goal
    INSIDE a <system-reminder> in that block ("A goal has been set: …"
    followed by the harness's plan boilerplate), so nothing inside the
    tag is stripped. None when the text carries no such tag."""
    m = _QUERY_RE.search(txt or "")
    if not m:
        return None
    return m.group(1).strip()


def _file(e: dict) -> str:
    a = _a(e)
    return str(e.get("file") or a.get("file_path") or a.get("path") or _s(e))


def _change(path: str, *, old: str | None = None, new: str = "",
            patch: str = "", edits: list | None = None):
    """One file change -> (patches, payload). The patch keeps the shape
    its harness gave it (old/new, new only, a codex patch block, a list
    of edits); the payload verdict is the same for all four."""
    body = "".join([old or "", new, patch]
                   + [str(x.get("old") or "") + str(x.get("new") or "")
                      for x in (edits or [])])
    if not body.strip():
        return None, "absent"
    p: dict = {"path": path}
    if patch:
        p["patch"] = patch
    elif edits is not None:
        p["edits"] = edits
    else:
        if old is not None:
            p["old"] = old
        p["new"] = new
    return [p], ("truncated" if _cut(body) else "carried")


def _apply_patches(cmd: str):
    """A codex command carrying `*** Begin Patch` -> (patches, payload):
    one patch per file section. The block rides either with real
    newlines (an apply_patch call, a heredoc) or as a JS string literal
    with `\\n` escapes (`const patch = "*** Begin Patch\\n..."` inside an
    exec cell); the sections are read from
    whichever form yields them."""
    out = []
    for body in (cmd, cmd.replace("\\n", "\n")):
        out = [{"path": path, "patch": f"*** {op} File: {path}\n{sect}"}
               for op, path, sect in PATCH_FILE_RE.findall(body)]
        if out:
            break
    if not out:
        return None, "absent"
    return out, ("truncated" if _cut(cmd) else "carried")


def _plain(e: dict):
    """A call that writes nothing, spelled `<name> <arguments>`: the
    whole args object when the dump carries one (uncapped), else the
    summary; never both, never the name twice (codex spells
    write_stdin's summary with the name in front).

    The verb is lowercased, like every other action line in the record
    (`cat f`, `edit f`, `todo t`, a raw shell line): the record spells
    actions shell-style, and which letters the harness capitalises in a
    tool's name carries nothing."""
    t, a = str(e.get("tool") or ""), _a(e)
    s = json.dumps(a, ensure_ascii=False) if a else _s(e)
    return (s if s.startswith(t) else f"{t.lower()} {s}".strip()), None, None


def _subagent(e):
    return f"subagent {_s(e)}", None, None


def _todo(e):
    return f"todo {_s(e)}", None, None


def _monitor(e):
    return f"monitor {_a(e).get('command') or _s(e)}", None, None


def _retrieve(verb):
    def r(e):
        return f"{verb} {_file(e)}", None, None
    return r


# claude-code and kimi-code speak the same vocabulary. Some dumps write a
# call twice — flat fields (file/old/new) and the raw `args` object — and
# either may hold the body (a Write only in args, an Edit only in the
# flat fields), so both are read.
def _c_bash(e):
    return str(_a(e).get("command") or _s(e)), None, None


def _c_edit(e):
    f, a = _file(e), _a(e)
    patches, pay = _change(f, old=str(e.get("old") or a.get("old_string") or ""),
                           new=str(e.get("new") or a.get("new_string") or ""))
    return f"edit {f}", patches, pay


def _c_write(e):
    f = _file(e)
    patches, pay = _change(f, new=str(e.get("new") or _a(e).get("content") or ""))
    return f"write {f}", patches, pay


CLAUDE = {
    "Bash": _c_bash, "Monitor": _monitor, "Read": _retrieve("cat"),
    "Edit": _c_edit, "Write": _c_write,
    "Agent": _subagent, "AgentSwarm": _subagent,
    **{n: _todo for n in ("TodoWrite", "TodoList", "TaskCreate", "TaskUpdate",
                          "TaskList", "TaskGet")},
    **{n: _plain for n in ("Glob", "Grep", "WebFetch", "WebSearch",
                           "NotebookEdit", "BashOutput", "KillShell",
                           "TaskOutput", "TaskStop", "ToolSearch",
                           "ScheduleWakeup", "SendMessage", "ListAgents",
                           "ReadMediaFile")},
}


# codex: exec cells ride raw (the JS wrapper or the shell line, as the
# dump spells it); a cell or an apply_patch call carrying a patch block
# yields one patch per file section.
def _x_exec(e):
    a, s = _a(e), _s(e)
    if e.get("tool") == "exec_command":           # exec_command: the shell line rides args.cmd
        s = str(a.get("cmd") or s)
    elif not s:                                   # no summary: the cell rides args
        s = str(a.get("input") or a.get("cmd") or "")
    if "*** Begin Patch" in s or "*** Begin Patch" in s.replace("\\n", "\n"):
        return (s, *_apply_patches(s))
    return s, None, None


def _x_wait(e):
    s = _s(e) or json.dumps(_a(e), ensure_ascii=False)
    return (s if s.lstrip().startswith("wait") else f"wait {s}"), None, None


def _x_patch(e):
    body = str(_a(e).get("input") or _s(e))
    return (body, *_apply_patches(body))


CODEX = {
    "exec": _x_exec, "exec_command": _x_exec,
    "wait": _x_wait, "wait_agent": _x_wait, "apply_patch": _x_patch,
    **{n: _plain for n in ("write_stdin", "spawn_agent", "send_message",
                           "list_agents", "followup_task", "create_goal")},
}


# grok-cli: search_replace bodies ride the flat old/new fields when the
# dump kept them (some releases dropped the call's arguments and its
# summary alike). write carries no flat
# body: the dump filed the call's argument JSON as its summary, whole
# when short, cut by the release past ~330 characters — the path is the
# first key and always whole, the content is whatever survived the cut.
def _g_replace(e):
    f = str(e.get("file") or _s(e))
    patches, pay = _change(f, old=str(e.get("old") or ""), new=str(e.get("new") or ""))
    return f"edit {f or '(path absent from dump)'}", patches, pay


def _json_string(body: str) -> str:
    """The text of a JSON string whose closing quote the release cut
    off: decoded as JSON when the cut left the escapes whole, else as
    the dump has it."""
    try:
        return json.loads('"' + body + '"')
    except ValueError:
        return body


def _g_write(e):
    f, new, s = str(e.get("file") or ""), str(e.get("new") or ""), _s(e)
    if not f and s.lstrip().startswith("{"):
        try:
            j = json.loads(s)
        except ValueError:
            j = None
        if isinstance(j, dict):
            f = str(j.get("file_path") or j.get("path") or "")
            new = new or str(j.get("content") or "")
        else:                                     # the cut JSON
            _, sep, rest = s.partition('"file_path": "')
            if sep:
                f, _, rest = rest.partition('"')
                _, sep, content = rest.partition('"content": "')
                new = new or (_json_string(content) if sep else "")
    elif not f:
        f = s
    patches, pay = _change(f, new=new)
    return f"write {f or '(path absent from dump)'}", patches, pay


GROK = {
    "run_terminal_command": lambda e: (_s(e), None, None),
    "read_file": _retrieve("cat"), "list_dir": _retrieve("ls"),
    "grep": _retrieve("grep"),
    "search_replace": _g_replace, "write": _g_write,
    "spawn_subagent": _subagent, "monitor": _monitor,
    "todo_write": _todo, "update_goal": _todo,
    "get_command_or_subagent_output": _plain, "kill_command_or_subagent": _plain,
}


# pi (glm, muse-spark): every payload rides args, whole.
def _p_bash(e):
    return str(_a(e).get("command") or _s(e)), None, None


def _p_edit(e):
    a = _a(e)
    p = str(a.get("path") or _s(e))
    eds = [{"old": str(x.get("oldText") or ""), "new": str(x.get("newText") or "")}
           for x in (a.get("edits") or [])]
    patches, pay = _change(p, edits=eds)
    return f"edit {p}", patches, pay


def _p_write(e):
    a = _a(e)
    p = str(a.get("path") or _s(e))
    patches, pay = _change(p, new=str(a.get("content") or ""))
    return f"write {p}", patches, pay


PI = {"bash": _p_bash, "read": _retrieve("cat"), "edit": _p_edit, "write": _p_write}


# qwen-code: the summary carries the whole call (a path, a shell line or
# a JSON body); edit and write bodies are ABSENT from the dump (the
# result shows the changed lines instead).
def _q_edit(verb):
    def r(e):
        s = _s(e)
        return f"{verb} {s or '(path absent from dump)'}", None, "absent"
    return r


QWEN = {
    "run_shell_command": lambda e: (_s(e), None, None),
    "read_file": _retrieve("cat"), "list_directory": _retrieve("ls"),
    "glob": _retrieve("grep"), "search_file_content": _retrieve("grep"),
    "edit": _q_edit("edit"), "write_file": _q_edit("write"),
    **{n: (lambda e, n=n: (f"todo {n} {_s(e)[:300]}", None, None))
       for n in ("get_goal", "update_goal", "record_artifact", "task_stop")},
}

# prime-agent (PrimeIntellect's own harness): ONE tool, an
# ipython cell; the code rides args.code, whole (summary = its head). What a
# cell does — run a training job, read a file, write one, dispatch a worker
# through rlm() — is written in the code the record shows; the tool's name
# decides its kind, so every cell is an execution and nothing is inferred
# from its text. The task prompt is the turn-0 user text, as in the codex dumps.
def _prime_ipython(e):
    return f"ipython {str(_a(e).get('code') or _s(e))}", None, None


PRIME = {"ipython": _prime_ipython}

DIALECTS = {"claude-code": CLAUDE, "kimi-code": CLAUDE, "codex": CODEX,
            "grok-cli": GROK, "pi": PI, "qwen-code": QWEN, "prime-agent": PRIME}


def _dialect(harness: str) -> tuple[str, dict]:
    """The table for a manifest harness name (a prefix match: the
    release spells variants like claude-code-x)."""
    for name in sorted(DIALECTS, key=len, reverse=True):
        if str(harness or "").startswith(name):
            return name, DIALECTS[name]
    raise UnknownHarness(f"{harness!r}: no dialect table; known: {sorted(DIALECTS)}")


def _render(e: dict, table: dict):
    t = str(e.get("tool") or "")
    fn = table.get(t)
    if fn is None:
        raise UnknownTool(
            f"{t!r} is not in this harness's table; the call: "
            + json.dumps({k: str(v)[:80] for k, v in e.items()}, ensure_ascii=False))
    return fn(e)

NO_RESULT = ("[no tool result in the dump: the turn ended before this call "
             "returned]")


def _obs_text(e: dict) -> str:
    """The observation as the agent saw it. Some harnesses wrap every
    tool result in a transport envelope ({"output": ...} / {"error":
    ...}, qwen-code): the envelope is the harness's, not the world's,
    and its escaped JSON is not what the agent read — unwrap it. Any
    other JSON result is world output and stays verbatim."""
    raw = str(e.get("text") or "")
    s = raw.lstrip()
    if not s.startswith("{"):
        return raw
    try:
        j = json.loads(raw)
    except Exception:
        return raw
    if isinstance(j, dict) and j and set(j) <= {"output", "error"}:
        return "\n".join(str(v) for v in (j.get("output"), j.get("error"))
                          if v not in (None, ""))
    return raw


def _walk(events: list, table: dict, sess, sub_tag: str | None = None,
          task: str | None = None):
    """One transcript -> canonical steps, each stamped with ts (when the
    result came back — a long call sits where it finished) and ts_call
    (when the call was made; a dispatch sorts by this one, so a worker's
    steps follow the call that started it even when the call blocked
    until the worker was done). sub_tag/task mark a worker's steps in
    the reasoning channel: every step says which worker acted and on
    what task.

    A tool_result answers the oldest open call of its own turn that it
    can name. The harness stamps call and result with the turn they
    belong to; a label names a call when it is the name of a tool the
    agent calls in this transcript; a label that names nothing the
    agent calls carries no name (grok-cli: null on every result; codex:
    the harness's own placeholder "tool" on a result it could not
    attribute) and answers the oldest open call of the turn under any
    name. Matching on the name alone would let one lost result shift
    every later result of that tool one call back, and on a forked codex
    conversation, whose contexts interleave in the dump, pair one
    context's polls with the other's answers. A result that finds no open call answers nothing the agent asked:
    it is text delivered to the agent at that position and rides the
    reasoning channel like any other message there (on codex it is the
    model's own turn commentary that the dump filed as a result, e.g. a
    remark that a job is still running), counted as results_without_call.
    A harness prompt (a user-role text: "continue", the goal re-sent)
    arriving while calls are still open means those calls never
    returned — the harness cut the turn — so each is closed as its own
    step with NO_RESULT as obs: the call was made, the dump holds no
    answer (pi: a long training command cut by the turn timeout);
    so is every call still open at the end.
    Each step then sits at its time: an answered call where its answer
    came back, an unanswered one where it was made; steps without a
    stamp keep their place among their neighbours."""
    steps: list[dict] = []
    pending: list[str] = []
    open_uses: list[dict] = []
    task_prompt = None
    task_from_tag = False
    first_marked = False
    first_user_seen = False
    names = {e.get("tool") for e in events if e.get("type") == "tool_use"}
    without_call = 0

    def mark(msg: str) -> str:
        nonlocal first_marked
        if sub_tag is None:
            return msg
        if not first_marked:
            first_marked = True
            head = (f"[worker {sub_tag} begins; task: "
                    f"{(task or '').strip()}]")
            return head + ("\n" + msg if msg else "")
        head = f"[worker {sub_tag} on task: {(task or '').strip()[:120]}]"
        return f"{head} {msg}" if msg else head

    def close(use: dict, e: dict | None, obs: str):
        cmd, patches, payload = _render(use, table)
        # the one-line description the agent writes on a call ("Launch
        # the learning-rate probe") is its own statement of what the call
        # is for — its words at that step, kept
        desc = str(use.get("desc") or "").strip()
        if desc:
            pending.append(desc)
        st = {"src": "agent", "msg": mark("\n".join(pending)),
              "tools": [{"fn": use.get("tool"), "cmd": cmd}],
              "obs": obs, "session": sess(e if e is not None else use),
              "api_turn": use.get("turn", (e or {}).get("turn")),
              "ts": (e or {}).get("ts") or use.get("ts"),
              "ts_call": use.get("ts") or (e or {}).get("ts")}
        if patches:
            st["patches"] = patches
        if payload:                    # a call that writes: the payload contract
            st["payload"] = payload
        if sub_tag is not None:
            st["actor"] = f"worker:{sub_tag}"
        pending.clear()
        steps.append(st)

    def flush_open():
        for u in list(open_uses):
            open_uses.remove(u)
            close(u, None, NO_RESULT)

    for e in sorted(events, key=lambda x: x.get("i", 0)):
        et = e.get("type")
        if et == "thinking":
            continue                       # excluded by policy
        if et == "text":
            txt = str(e.get("text") or "")
            if e.get("role") == "assistant":
                if txt.strip():
                    pending.append(txt)
                continue
            if open_uses and txt.strip():  # harness prompt while calls
                flush_open()               # are open: they never returned
            # every message the agent received is a step of its own —
            # the harness's words, kept apart from the agent's: a
            # nudge, the goal re-sent, a context-compaction handoff
            # summary (the agent's carried memory), a worker's
            # completion notice carrying its report (codex
            # subagent_notification: the numbers the parent then
            # reasons from). A worker's
            # FIRST message is the parent's instruction and rides the
            # dispatch step / opening marker instead.
            if sub_tag is None:
                # THE SHIPPED TASK. A harness that TAGS the user's query
                # (grok-cli wraps it in <user_query>…</user_query>, with
                # <system-reminder> boilerplate inside) names the task by
                # that tag, and the tag wins over any untagged user text:
                # grok-cli sends a bare handshake ("Reply with exactly:
                # OK") and <user_info> environment blocks before the
                # query, which must not be taken as the task. Untagged
                # dumps take the first user text not starting with "<".
                q = _tagged_query(txt)
                if q is not None:
                    if not task_from_tag:
                        task_prompt, task_from_tag = q, True
                elif task_prompt is None and txt.strip() \
                        and not txt.lstrip().startswith("<"):
                    task_prompt = txt
            elif not first_user_seen and e.get("role") == "user":
                first_user_seen = True     # the instruction, carried elsewhere
                continue
            if txt.strip():
                st = {"src": "user" if e.get("role") == "user"
                      else "system", "msg": txt, "tools": None,
                      "obs": None, "session": sess(e),
                      "api_turn": e.get("turn"), "ts": e.get("ts")}
                if sub_tag is not None:
                    st["actor"] = f"worker:{sub_tag}"
                steps.append(st)
            continue
        if et == "tool_use":
            open_uses.append(e)
            continue
        if et == "tool_result":
            label = e.get("tool")
            use = next((u for u in open_uses
                        if u.get("turn") == e.get("turn")
                        and (label not in names
                             or u.get("tool") == label)), None)
            obs = _obs_text(e)
            if use is None:                # answers nothing the agent asked
                without_call += 1
                if obs.strip():
                    pending.append(obs)
                continue
            open_uses.remove(use)
            if e.get("is_error"):
                obs = "[tool error]\n" + obs
            close(use, e, obs)
    for u in list(open_uses):              # calls still open at the end
        open_uses.remove(u)
        close(u, None, NO_RESULT)
    if pending:                            # trailing message (child: the
        st = {"src": "agent",              # report it authored)
              "msg": mark("\n".join(pending)), "tools": None,
              "obs": None,
              "session": steps[-1]["session"] if steps else 0,
              "api_turn": events[-1].get("turn") if events else None,
              "ts": events[-1].get("ts") if events else None}
        if sub_tag is not None:
            st["actor"] = f"worker:{sub_tag}"
        steps.append(st)
    last = ""                              # stable: unstamped steps keep
    keyed = []                             # their place among neighbours
    for s in steps:
        last = s.get("ts") or last
        keyed.append((last, s))
    steps = [s for _, s in sorted(keyed, key=lambda x: x[0])]
    return steps, task_prompt, without_call


def _drop_replays(steps: list) -> tuple[list, int]:
    """Drop transcript replays (base_adapter.replay_steps: a session
    resume that re-emits the old transcript as if new, log uuids
    included). A step is identified by its (src, command, observation,
    message), byte for byte; a replayed block starts at an agent tool
    step. The next agent step after a dropped block says so in its
    reasoning channel."""
    keys = [(s.get("src"),
             s["tools"][0]["cmd"] if s.get("tools") else None,
             s.get("obs"), s.get("msg")) for s in steps]
    drop = _base.replay_steps(
        keys, lambda i: steps[i].get("src") == "agent" and bool(keys[i][1]))
    if not drop:
        return steps, 0
    out = []
    note = None
    for i, s in enumerate(steps):
        if i in drop:
            note = (note or 0) + 1
            continue
        if note and s.get("src") == "agent":     # the next agent step
            s = dict(s)                          # carries the note
            s["msg"] = (f"[transcript replay: {note} steps identical to "
                        f"earlier steps of this record were dropped here]"
                        + ("\n" + s["msg"] if s.get("msg") else ""))
            note = None
        out.append(s)
    return out, len(drop)


def _tokens(text: str) -> set:
    return {t for t in re.split(r"[^A-Za-z0-9.]+", text.lower()) if len(t) > 1}


def _first_user_text(events: list) -> str:
    for e in sorted(events, key=lambda x: x.get("i", 0)):
        if e.get("type") == "text" and e.get("role") == "user" \
                and str(e.get("text") or "").strip():
            return str(e["text"])
    return ""


def _worker_tags(ids: list) -> dict:
    """A worker's tag is what tells it from its siblings: its id with
    the prefix all the run's workers share removed, cut to the shortest
    length of at least 8 characters at which every worker's tag is
    distinct. A fixed-length id prefix is not enough: on harnesses whose
    ids share a long prefix (codex 00000000-0000-4000-8000-0000000000NN;
    kimi agent-10/agent-100; grok-cli time-ordered ids) every worker
    would get the same tag and different workers would fold into one
    actor."""
    if not ids:
        return {}
    common = os.path.commonprefix(ids) if len(ids) > 1 else ""
    rem = {cid: cid[len(common):] for cid in ids}
    longest = max(len(r) for r in rem.values())
    ln = min(8, longest)
    while ln < longest and len({r[:ln] for r in rem.values()}) < len(ids):
        ln += 1
    return {cid: r[:ln] for cid, r in rem.items()}


def _inline_children(steps: list, sub: dict, table: dict, sess) -> dict:
    """Place each worker's steps in the parent's sequence and mark the
    parent's DISPATCH step (the call that started the worker) and its
    RELAY steps (later calls whose result carries the worker's report or
    status back).

    Linking: the worker's id (its key in the dump) appears in the
    dispatch step's observation on harnesses that echo it (claude-code
    agentId, kimi agent_id, codex spawn output); harnesses that do not
    (grok-cli: a harness-side id the dump rewrote) link by task text —
    the dispatch command's words against every worker's own task
    prompt, each word weighted by how rare it is across the workers'
    prompts (a run label or a number decides, "run"/"screen" does not);
    best unused worker at >= 0.5 of the dispatch's weight. Relays: later
    parent steps whose observation names the worker's id or an id the
    dispatch observation introduced (a token no earlier step carried).

    Placement: by timestamp when both sides carry one; otherwise right
    after the dispatch step; a worker with neither is placed among its
    placed siblings by id order when worker ids are time-ordered (they
    are on the harnesses that lack timestamps) — before the first
    placed sibling's dispatch, after the last one's, or between the
    two nearest — and counted as unplaced-by-order (some dumps' parent
    transcript covers only the run's tail, so sorting such workers after
    the whole parent transcript would put them on the wrong clock)."""
    n = {"inlined": 0, "inlined_steps": 0, "unplaced": 0, "by_ts": 0,
         "by_dispatch": 0, "by_id_order": 0, "relays": 0,
         "results_without_call": 0}
    parent = list(steps)
    children = sub.get("events") or {}
    if not children:
        return n
    tags = _worker_tags(list(children))
    tool_steps = [i for i, s in enumerate(parent)
                  if s.get("src") == "agent" and s.get("tools")]
    harness_steps = [i for i, s in enumerate(parent)
                     if s.get("src") in ("user", "system")]
    disp_cands = [i for i in tool_steps
                  if parent[i]["tools"][0]["cmd"].startswith("subagent ")]
    tasks = {cid: _first_user_text(cev) for cid, cev in children.items()}
    df = Counter()
    for t in tasks.values():
        df.update(_tokens(t))
    nk = max(1, len(tasks))

    def weight(tok):
        return math.log(1 + nk / (1 + df.get(tok, 0)))

    def text_of(i):
        s = parent[i]
        return " ".join([str(s.get("msg") or ""), str(s.get("obs") or ""),
                         s["tools"][0]["cmd"] if s.get("tools") else ""])

    used: set = set()
    placed: dict = {}      # cid -> dispatch index (by-dispatch placement)
    pending: list = []     # (cid, c_steps) with no ts and no dispatch
    ts_blocks: list = []
    resolved: list = []    # (cid, tag, di, c_steps)
    for cid in sorted(children):
        cev = children[cid]
        tag = tags[cid]
        task_text = tasks[cid]
        # the dump may keep a truncated handle (a prefix of the hex
        # id): a long hex-like key matches as a prefix of the echoed
        # id; a short key (kimi agent-0) must match whole
        cid_re = (r"(?<![\w-])" + re.escape(cid)
                  + (r"[0-9a-f]*" if len(cid) >= 12 else r"(?![\w-])"))
        di = next((i for i in tool_steps
                   if i not in used and re.search(
                       cid_re, str(parent[i].get("obs") or ""))), None)
        if di is None and task_text:
            ttoks = _tokens(task_text)
            best, best_i = 0.0, None
            for i in disp_cands:
                if i in used:
                    continue
                ctoks = _tokens(parent[i]["tools"][0]["cmd"][9:])
                tot = sum(weight(t) for t in ctoks)
                if not tot:
                    continue
                sc = sum(weight(t) for t in ctoks & ttoks) / tot
                if sc > best:
                    best, best_i = sc, i
            if best >= 0.5:
                di = best_i
        # the worker's label on its own steps: the dispatch call's
        # summary when a dispatch step is linked (the full instruction
        # then rides that step, below), else the instruction itself —
        # the parent's words to the worker exist nowhere else
        task = None
        if di is not None:
            used.add(di)
            cmd = parent[di]["tools"][0]["cmd"]
            task = cmd[9:] if cmd.startswith("subagent ") else None
        task = task or task_text or str(
            next((it.get("label") for it in sub.get("index", [])
                  if it.get("id") == cid), "") or "")
        c_steps, _, orphans = _walk(cev, table, sess, sub_tag=tag, task=task)
        n["results_without_call"] += orphans
        # the worker's steps: its own acts and messages, and the harness
        # messages it received after its instruction
        if not c_steps:
            continue
        has_ts = all(s.get("ts") for s in c_steps) and any(
            s.get("ts") for s in parent)
        if di is None and not has_ts:
            pending.append((cid, c_steps))
            continue
        n["inlined"] += 1
        n["inlined_steps"] += len(c_steps)
        resolved.append((cid, tag, di, c_steps))
        if has_ts:
            n["by_ts"] += 1
            ts_blocks.append(c_steps)
        else:
            n["by_dispatch"] += 1
            placed[cid] = c_steps
            placed[cid + "@"] = di
    # marking: dispatch, then the harness-side ids each asynchronous
    # acknowledgement introduces (a token no earlier step carried and no
    # other dispatch's acknowledgement carries — session paths and run
    # ids recur in every acknowledgement, a worker's handle in one), then
    # the relays addressed by those ids
    ack_tokens: Counter = Counter()
    per_disp: dict = {}
    for cid, tag, di, c_steps in resolved:
        if di is None:
            continue
        d = parent[di]
        obs = str(d.get("obs") or "")
        report = next((str(s.get("msg") or "").split("]", 1)[-1].strip()
                       for s in reversed(c_steps)
                       if not s.get("tools") and s.get("src") == "agent"), "")
        sync = bool(report) and report[:120] in obs
        toks = set()
        if not sync:
            before = " ".join(text_of(i) for i in range(di))
            toks = {t for t in re.findall(r"[A-Za-z0-9][\w-]{11,}", obs)
                    if any(ch.isdigit() for ch in t)
                    and t not in d["tools"][0]["cmd"] and t not in before}
        ack_tokens.update(toks)
        per_disp[cid] = (sync, toks)
    for cid, tag, di, c_steps in resolved:
        if di is None:
            continue
        d = parent[di]
        sync, toks = per_disp[cid]
        d["delegation"] = "dispatch+relay" if sync else "dispatch"
        d["ts"] = d.get("ts_call") or d.get("ts")
        # the instruction the parent wrote for the worker is the
        # parent's own words at this step. The release scrubbed it from
        # the dispatch call's arguments; it survives as the worker's
        # first user message (claude-code: the Task prompt)
        instr = tasks[cid].strip()
        d["msg"] = (f"[dispatch -> worker {tag}; its own steps are "
                    f"inlined at their own time, marked [worker {tag}]]"
                    + (f"\n[instruction to worker {tag}]\n{instr}" if instr else "")
                    + ("\n" + d["msg"] if d.get("msg") else ""))
        pats = [r"(?<![\w-])" + re.escape(cid)
                + (r"[0-9a-f]*" if len(cid) >= 12 else r"(?![\w-])")]
        pats += [r"(?<![\w-])" + re.escape(t) + r"(?![\w-])"
                 for t in toks if ack_tokens[t] == 1]
        for i in sorted(set(tool_steps) | set(harness_steps)):
            if i <= di or i in used:
                continue
            s = parent[i]
            # a relay is a step whose incoming text names the worker: a
            # call's result, or a harness message delivered to the
            # parent (codex subagent_notification carrying the report)
            o = str(s.get("obs") or "") if s.get("tools") else str(s.get("msg") or "")
            if any(re.search(pt, o) for pt in pats):
                if s.get("delegation") != "relay":
                    s["delegation"] = "relay"
                    note = (f"[relay <- worker {tag}: its report or "
                            f"status; the worker's own steps are in "
                            f"the record]")
                    if s.get("tools"):
                        s["msg"] = note + ("\n" + s["msg"] if s.get("msg") else "")
                    else:
                        s["note"] = note
                    n["relays"] += 1
    # placement
    after: dict = {}
    for cid, c_steps in placed.items():
        if cid.endswith("@"):
            continue
        after.setdefault(placed[cid + "@"], []).extend(c_steps)
    if pending:
        anchors = sorted((cid, placed[cid + "@"]) for cid in placed
                         if not cid.endswith("@"))
        if not anchors:
            n["unplaced"] += len(pending)
        else:
            for cid, c_steps in pending:
                n["inlined"] += 1
                n["inlined_steps"] += len(c_steps)
                n["by_id_order"] += 1
                prev = [a for a in anchors if a[0] < cid]
                if prev:
                    # after the previous sibling's dispatch block
                    after.setdefault(prev[-1][1], []).extend(c_steps)
                else:
                    after.setdefault(-1, []).extend(c_steps)
    if after:
        merged = list(after.get(-1, []))
        for i, s in enumerate(parent):
            merged.append(s)
            merged.extend(after.get(i, []))
        parent = merged
    for c_steps in ts_blocks:
        parent.extend(c_steps)
    if ts_blocks:
        # stable sort by timestamp; steps without one keep their place
        # relative to their neighbours (key = the last stamp seen)
        last = ""
        keyed = []
        for s in parent:
            last = s.get("ts") or last
            keyed.append((last, s))
        parent = [s for _, s in sorted(keyed, key=lambda x: x[0])]
    steps[:] = parent
    return n


def _emit_usage(events: list) -> dict | None:
    """usage.json from the dump's own per-call usage stamps (codex:
    every assistant event carries {in, out, cache_read, reasoning_out};
    multiple events per harness turn = separate API calls, so per-turn
    sums are the true account). thinking_tokens = reasoning_out, exact;
    the non-reasoning remainder splits into text/action by visible char
    proportion. Runs with no usage stamps (claude-code: redacted
    thinking, no numbers) return None and get the run-total account
    (_emit_usage_armtotal) instead; no per-call char proxy. Inlined
    worker steps keep their own api_turn numbers, which can coincide
    with parent turns; usage is built from PARENT events only, so a
    worker step joined by api_turn reads a parent turn's account (a
    small share of steps)."""
    per: dict = {}
    for e in events:
        t = e.get("turn")
        u = e.get("usage")
        if t is None:
            continue
        row = per.setdefault(str(t), {"out": 0, "reasoning": 0,
                                      "prose_chars": 0,
                                      "tool_chars": 0})
        if u:
            row["out"] += int(u.get("out") or 0)
            row["reasoning"] += int(u.get("reasoning_out") or 0)
        if e.get("type") == "text" and e.get("role") == "assistant":
            row["prose_chars"] += len(str(e.get("text") or ""))
        elif e.get("type") == "tool_use":
            row["tool_chars"] += (len(str(e.get("summary") or ""))
                                  + len(str(e.get("old") or ""))
                                  + len(str(e.get("new") or "")))
    if not any(r["out"] for r in per.values()):
        return None      # no usage stamps: see _emit_usage_armtotal
    # when the dump carries NO reasoning stamps, hidden thinking per turn
    # is the RESIDUAL out - visible_chars / CPT, the same rule as the
    # run-total band, at its calibrated end; a plain char split would
    # read hidden thinking as text/action.
    residual = not any(r["reasoning"] for r in per.values())
    CPT = _base.CPT
    turns = {}
    for t, r in per.items():
        if not r["out"]:
            continue
        tot = r["prose_chars"] + r["tool_chars"]
        if residual:
            hidden = max(0.0, r["out"] - tot / CPT)
            r["reasoning"] = int(round(hidden))
        nonr = max(0, r["out"] - r["reasoning"])
        text = (nonr * r["prose_chars"] / tot) if tot else float(nonr)
        turns[t] = {"out": r["out"],
                    "thinking_tokens": r["reasoning"],
                    "text_tokens": round(text),
                    "action_tokens": round(nonr - text),
                    "prose_chars": r["prose_chars"],
                    "tool_chars": r["tool_chars"]}
    return {"grain": "api_call",
            "reporter": ("speedrun sanitized dump per-call usage "
                         "(in/out/cache_read" + ("" if residual else "/reasoning_out") + "); "
                         + (f"no reasoning stamps: thinking = per-turn residual out - visible_chars/{CPT} "
                            "(run-total band rule, calibrated end); " if residual else
                            "exact reasoning_out as thinking_tokens; ")
                         + "non-reasoning remainder char-split into text/action; parent events "
                         "only (an inlined worker step's api_turn is the worker's own turn number)"),
            "turns": turns}


def _emit_usage_armtotal(events: list, sub_events: list,
                         rmeta: dict) -> dict | None:
    """Run-total usage for runs without usage stamps (claude-code:
    thinking redacted, no per-call stamps). The run's own reported
    out_tok (manifest economics) is exact; the visible channel (assistant
    prose + agent-authored commands/patches, workers included) is
    countable; the RESIDUAL is the invisible thinking budget, banded by
    the chars-per-token ratio CPT_BAND: 2.9 (calibrated on a run with
    true usage: visible chars / (out - reasoning_out)) to 4.2 (generic
    English prose). A run-total account (grain arm_total) never feeds
    T1's per-belief windows (conclusion_budget_doing reads api_call
    grain only); T1's arm_budget_doing reads the band."""
    econ = rmeta.get("economics") or {}
    out_tok = econ.get("out_tok")
    if not out_tok:
        return None
    allev = list(events) + list(sub_events)
    prose = sum(len(str(e.get("text") or "")) for e in allev
                if e.get("type") == "text"
                and e.get("role") == "assistant")
    tool = sum(len(str(e.get("summary") or ""))
               + len(str(e.get("old") or ""))
               + len(str(e.get("new") or "")) for e in allev
               if e.get("type") == "tool_use")
    band = {}
    for cpt in _base.CPT_BAND:
        vis = (prose + tool) / cpt
        band[str(cpt)] = {
            "visible_tokens": round(vis),
            "thinking_tokens_residual": round(out_tok - vis),
            "think_side_share": round((out_tok - tool / cpt)
                                      / out_tok, 2),
            "do_side_share": round(tool / cpt / out_tok, 2)}
    return {"grain": "arm_total",
            "reporter": "manifest economics out_tok + visible-channel"
                        " chars; residual = invisible (redacted) "
                        "thinking, banded chars/token 2.9 (calibrated "
                        "on a run with exact token counts) .. 4.2 "
                        "(generic); the directional verdict is stable "
                        "across the band",
            "out_tok": out_tok, "prose_chars": prose,
            "tool_chars": tool, "band": band}


def _payload_chars(e: dict) -> int:
    """Everything of the model's own output this event carries, wherever
    the dialect puts it: prose, thinking, the call's summary, the flat
    edit fields, and the args object."""
    n = len(str(e.get("text") or ""))
    if e.get("type") == "tool_use":
        # the same call is often written twice — flat fields AND args —
        # so the payload is the larger of the two, never their sum
        flat = (len(str(e.get("summary") or "")) + len(str(e.get("old") or ""))
                + len(str(e.get("new") or "")))
        a = e.get("args")
        args = len(json.dumps(a, ensure_ascii=False)) if isinstance(a, dict) else 0
        n += max(flat, args)
    return n


def _carried_chars(s: dict) -> int:
    n = len(str(s.get("msg") or "")) + len(str(s.get("obs") or ""))
    for t_ in (s.get("tools") or []):
        n += len(str(t_.get("cmd") or ""))
    for p in (s.get("patches") or []):
        n += sum(len(str(v)) for k, v in p.items() if k != "path")
        n += sum(len(str(x.get("old") or "")) + len(str(x.get("new") or ""))
                 for x in (p.get("edits") or []))
    return n


def _coverage(events: list, sub_events: list, steps: list) -> dict:
    """How much of the dump the record carries. A dialect bug loses payloads silently:
    reading the wrong field yields an empty string, not an error. So the
    converter states how much of the dump it carried: raw payload
    characters in, record characters out. Well under 1.0 means a dialect
    is being read wrong (thinking, excluded by policy, is counted
    separately and not expected in the record)."""
    raw = sum(_payload_chars(e) for e in list(events) + list(sub_events)
              if e.get("type") != "thinking")
    think = sum(len(str(e.get("text") or "")) for e in list(events) + list(sub_events)
                if e.get("type") == "thinking")
    got = sum(_carried_chars(s) for s in steps)
    return {"raw_payload_chars": raw, "record_chars": got,
            "coverage": round(got / raw, 3) if raw else None,
            "thinking_chars_excluded_by_policy": think}


def _defects(events: list, sub_events: list, steps: list,
             n_results_without_call: int, n_replay: int) -> dict:
    """What this dump does not carry, counted, so a reader sees the gap
    as a number instead of inferring it from silence.
    payloads: the payload contract's tally over every call that
    writes, by tool. upstream_cuts: fields the release itself cut
    ("…[+N chars]"), by event type and field — results, the agent's
    prose, call summaries, edit bodies; the text rides cut, marker in
    place. calls_without_result: the harness cut the turn (a prompt
    arrived while the call was open) or the dump ends before the
    answer. None of these is the converter's to repair: they are
    facts about the release, declared."""
    cuts: Counter = Counter()
    for e in list(events) + list(sub_events):
        for k in ("text", "summary", "old", "new", "desc"):
            if _cut(e.get(k)):
                cuts[f"{e.get('type')}.{k}"] += 1
        a = e.get("args")
        if isinstance(a, dict) and _cut(json.dumps(a, ensure_ascii=False)):
            cuts["tool_use.args"] += 1
    pay: dict = {"carried": Counter(), "truncated": Counter(), "absent": Counter()}
    for s in steps:
        if s.get("payload") and s.get("tools"):
            pay[s["payload"]][s["tools"][0]["fn"]] += 1
    return {
        "payloads": {k: {"n": sum(c.values()), "by_tool": dict(c)}
                     for k, c in pay.items()},
        "upstream_cuts": dict(sorted(cuts.items())),
        "calls_without_result": sum(1 for s in steps if s.get("obs") == NO_RESULT),
        "results_without_call": n_results_without_call,
        "replay_steps_dropped": n_replay,
        "reading": "payload absent/truncated = the dump kept no (whole) "
                   "body for a call that writes; the step says THAT the "
                   "file changed and the observation says what came back. "
                   "upstream_cuts = fields the release truncated; the text "
                   "rides cut, marker in place. Nothing here is repaired "
                   "by the converter.",
    }


def convert(traces: Path, run_id: str, out_root: Path) -> Path:
    ev = _load(traces / f"events-{run_id}.json.gz")
    man = _load(traces / "manifest.json.gz")
    rmeta = next(r for r in man["runs"] if r["run"] == run_id)
    dialect, table = _dialect(rmeta.get("harness", ""))

    sess_ids: dict = {}

    def sess(e):
        sid = e.get("session_id")
        if sid is None:
            return 0
        return sess_ids.setdefault(sid, len(sess_ids))

    steps, task_prompt, n_without_call = _walk(ev["events"], table, sess)
    steps, n_replay = _drop_replays(steps)

    # inline worker (subagent) transcripts, with dispatch/relay marking
    sub_events: list = []
    link = {"inlined": 0, "inlined_steps": 0, "unplaced": 0}
    sub_path = traces / f"subagents-{run_id}.json.gz"
    if sub_path.exists():
        sub = _load(sub_path)
        for cev in (sub.get("events") or {}).values():
            sub_events.extend(cev)
        link = _inline_children(steps, sub, table, sess)

    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "steps.json").write_text(
        json.dumps(steps, ensure_ascii=False))
    usage = _emit_usage(ev["events"])
    if usage is None:
        usage = _emit_usage_armtotal(ev["events"], sub_events, rmeta)
    if usage is not None:
        (run_dir / "usage.json").write_text(
            json.dumps(usage, ensure_ascii=False))
    (run_dir / "meta.json").write_text(json.dumps({
        "task_name": f"nanogpt-speedrun-{rmeta.get('track', 'track3')}",
        "model": rmeta.get("model_id"),
        "agent": rmeta.get("harness"),
        "dialect": dialect,
        "trial_name": run_id,
        "reward": rmeta.get("best_record"),
        "reward_note": "best validated record in TRAIN STEPS - lower is "
                       "better (feedback_polarity lower_better); baseline "
                       f"{rmeta.get('baseline')}",
        "duration_seconds": int(float(rmeta.get("agent_h", 0)) * 3600),
        "hit_budget": False,
        "n_subagents_inlined": link["inlined"],
        "n_subagents_unplaced": link["unplaced"],
        "subagent_linking": link,
        "n_replay_steps_dropped": n_replay,
        "n_calls_without_result": sum(1 for s in steps
                                      if s.get("obs") == NO_RESULT),
        "n_results_without_call": n_without_call
        + link.get("results_without_call", 0),
        "task_prompt": task_prompt or "(goal message absent from dump)",
        "dump_coverage": _coverage(ev["events"], sub_events, steps),
        "dump_defects": _defects(ev["events"], sub_events, steps,
                                 n_without_call + link.get("results_without_call", 0),
                                 n_replay),
    }, ensure_ascii=False, indent=1))
    n_ag = sum(1 for s in steps if s["src"] == "agent")
    pay = Counter(s.get("payload") for s in steps if s.get("payload"))
    print(f"{run_id} [{dialect}]: {len(steps)} steps ({n_ag} agent, "
          f"{link['inlined_steps']} from {link['inlined']} inlined workers, "
          f"{link['unplaced']} workers unplaced, {n_replay} replay steps "
          f"dropped; payloads carried {pay['carried']} / truncated "
          f"{pay['truncated']} / absent {pay['absent']}) -> {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces_dir")
    ap.add_argument("run_id")
    ap.add_argument("--out", default="data/speedrun/runs")
    a = ap.parse_args()
    convert(Path(a.traces_dir), a.run_id, Path(a.out))


if __name__ == "__main__":
    main()
