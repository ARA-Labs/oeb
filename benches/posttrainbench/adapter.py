"""L0 adapter: PostTrainBench run dir -> canonical scope-tagged record.

A run is an open-ended post-training episode (PostTrainBench,
aisa-group/PostTrainBench-Trajectories): the agent gets one H100 and a
10h budget to fine-tune a small base LM so it maximizes a held-out
benchmark (gsm8k / aime / bfcl / ...). The raw log is the harness
stdout `solve_out.txt`, one of four CLI dialects (each line may carry a
`[ISO-timestamp] ` prefix, stripped before JSON parse):

  claude stream-json    claude-code / glmx (glm runs the claude CLI)
  codex event stream    codex CLI (`/bin/bash -lc '...'` commands)
  cursor-cli stream     tool_call events (started / completed) whose body
                        is one `<name>ToolCall` object
  opencode event stream step_start / text / tool_use with a `part`
                        payload; tool results are inline in part.state

SCOPE (shared with the base, speedrun and chipbench adapters). Whether a
step observed the container is a fact the harness knows: which KIND of
tool it ran. A shell call and a poll of a running
process observe the container (env); a file read, an edit, a delegation
and bookkeeping do not (workspace). The tool's NAME decides — never the
text of the command. What a shell call showed or launched (a `sed -n` of
evaluate.py, a `python train.py`) is the extractor's reading of its
action and observation; guessing it from the command
string misreads e.g. `sed -n '1,260p' evaluate.py` as a training run. A
tool name the shared table (benches/base_adapter.py TOOL_KIND) does not
know stops the adapter, and the name is added to the table by hand.

  claude-code   tool_use.name           -> TOOL_KIND
  codex         item.type               command_execution = shell
                                        (empty command = poll),
                                        file_change = edit,
                                        mcp_tool_call = TOOL_KIND[tool]
  cursor-cli    <name>ToolCall          -> TOOL_KIND
  opencode      part.tool               -> TOOL_KIND

THE RECORD CARRIES (the same fields as the speedrun converter):
  tool_kind     on every step (shell | poll | read | edit | delegate |
                bookkeeping; the submission step is `message`); the
                extractor's `_channel()` (oeb/graph/extract.py)
                reads it as a fact.
  patches       on every edit step: the action behind the edit (old ->
                new), so the judges read WHAT changed.
                claude Edit/Write and opencode edit/write carry the
                payload; codex file_change carries only {path, kind} —
                those ride as edits without payload, counted.
  api_turn      on every claude-code / opencode step (one turn per
                message id). usage.json: opencode stamps real per-message
                tokens (input/output/reasoning) -> grain api_call;
                claude-code's per-message stamps are first-block
                snapshots (output_tokens 3..7, a small fraction of the
                run's output) and thinking is redacted -> the closing
                result line's run total, grain arm_total, thinking =
                residual (as the speedrun converter does for claude-code);
                codex has one turn.completed usage line with exact
                reasoning tokens -> arm_total; cursor-cli likewise one
                closing total -> arm_total.
  delegation    on claude Task/Agent steps ("dispatch+relay"): the call
                started a worker and its result is the worker's WRITTEN
                report (same-model prose, not first-hand world output).
                When the dump carries the worker's own events they are
                inlined at their position and actor-marked; otherwise
                the call is counted as without transcript. The
                harness's task_started / task_progress lines ride the
                step's reasoning channel as [worker ... progress: ...].
  NO_RESULT     as obs on a call the dump never answers (a harness
                prompt while calls are open cuts the turn, as in the
                speedrun converter); the manifest counts them.
  record_gaps   in the manifest: edits without payload, calls without
                result, subagent calls without transcript, replay steps
                dropped.

Scope rules (recorded in the manifest): the MODEL-UNDER-TRAINING and
its training/eval runs ARE the bench world; behavioral evidence arrives
both from launching runs and from observing them.
  shell call (any command)                        -> env (kind exec)
  poll of a running process (TaskOutput,
  BashOutput, Monitor; codex empty command)       -> env (kind readlog)
  file reads/edits/writes, delegations,
  task-list bookkeeping                           -> workspace
  final message at time-out (checkpoint already
  saved; bench-side eval is post-hoc and NEVER
  enters the record)                              -> submission

world.json is written here from code, as the base / speedrun / chipbench
adapters write theirs (the bench DECLARES the world's shape, the scorer
never guesses it, an undeclared reading abstains).
It says: every PostTrainBench target is an accuracy / win-rate
(higher_better); the agent's own work is recorded in the patch channel
(shell redirections are not read); the provided evaluate.py is graded
material the agent can run in-run (task_given); no verdict arrives
mid-run (external False). assemble reads authored_channel to build
the artifact plane from the record's patches and feedback_polarity for
the outcome block; the coverage jury reads submission and measure_size;
the measure layer never reads world.json.

Usage:
  python benches/posttrainbench/adapter.py <run_dir> <out_dir>
<run_dir> holds solve_out.txt (+ SOURCE.txt naming the HF source run).
Writes <out_dir>/record.jsonl, record_manifest.json, world.json and,
when the dialect stamps tokens, usage.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the oeb package
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_prompt  # noqa: E402  (benches/posttrainbench/task_prompt.py)
from oeb.record.harness_streams import result_text, sniff_codex  # noqa: E402

# the ONE tool-kind table, shared with the base / speedrun / chipbench adapters —
# imported, not copied, so the benches cannot drift apart
_BASE = Path(__file__).resolve().parent.parent / "base_adapter.py"
_spec = importlib.util.spec_from_file_location("base_adapter", _BASE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
tool_kind, CPT, CPT_BAND = _base.tool_kind, _base.CPT, _base.CPT_BAND
ENV_KINDS = ("shell", "poll")

TS_PREFIX = re.compile(r"^\[[0-9TZ:.\-]+\]\s+")
NO_RESULT = ("[no tool result in the dump: the turn ended before this call "
             "returned]")


def _txt(v):
    """A field as text, whole (the record keeps full text, as in benches/base_adapter.py)."""
    if v is None:
        return ""
    if isinstance(v, str):
        s = v
    else:
        try:
            s = json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            s = str(v)
    return s


def _exec_struct(cmd: str, kind: str) -> dict:
    head = cmd.strip().split()[0] if cmd.strip() else "(poll)"
    targets = re.findall(r"[\w./-]+\.(?:py|sh|log|out|json)", cmd)[:4]
    return {"kind": kind, "prog": head[:40],
            "target": " ".join(targets)}


def _unwrap_codex(cmd: str) -> str:
    m = re.match(r"^\s*(?:/\w\S*/)?(?:ba)?sh\s+-l?c\s+(.*)$", cmd, re.S)
    if not m:
        return cmd
    body = m.group(1).strip()
    if len(body) >= 2 and body[0] in "'\"" and body[-1] == body[0]:
        body = body[1:-1]
    return body


def _edit_patches(name: str, inp: dict) -> list[dict] | None:
    """The action behind an edit tool call, in the record's patch shape
    ({path, old?, new}). Key names differ by harness; the payload is
    whatever the call carried — absent payload rides as an empty new."""
    if not isinstance(inp, dict):
        return None
    path = (inp.get("file_path") or inp.get("filePath")
            or inp.get("notebook_path") or inp.get("path") or "")
    old = inp.get("old_string") if "old_string" in inp else inp.get("oldString")
    new = (inp.get("new_string") if "new_string" in inp
           else inp.get("newString") if "newString" in inp
           else inp.get("new_source") if "new_source" in inp
           else inp.get("content"))
    if path == "" and old is None and new is None:
        return None
    p = {"path": str(path), "new": "" if new is None else str(new)}
    if old is not None:
        p["old"] = str(old)
    return [p]


def world_declaration(target: str) -> dict:
    """world.json — the bench DECLARES the world's shape (the scorer
    never guesses it; an undeclared reading abstains).
    The fields the base / speedrun / chipbench adapters write, plus
    submission and measure_size for the coverage jury. Every
    PostTrainBench target is an accuracy / win-rate (higher_better);
    the agent's own work is recorded in the patch channel; the provided
    evaluate.py is graded material the
    agent can run in-run (task_given); no verdict arrives mid-run
    (external False) — the held-out grade is post-hoc and stays out."""
    return {
        "scope_default": "agent",
        "feedback_polarity": "higher_better",
        "authored_channel": {"kind": "patches+shell_redirect"},
        "action_genre": "dev_shell",
        "in_run_authority": {"task_given": True, "external": False},
        "declared_by": "benches/posttrainbench/adapter.py",
        # the SUBMISSION path the measure is applied to: the coverage
        # jury's promotion acts are the writes into it
        "submission": "final_model",
        # the size of each benchmark's published test set: aime2025 = the
        # 30 AIME 2025 problems (inspect_evals/aime2025), healthbench = the
        # 245 rows of evaluation_code/data/healthbench.jsonl,
        # arenahardwriting = the 250 prompts of arena-hard-v2.0/question.jsonl
        # (creative-writing subset), gpqamain = the 448 questions of GPQA
        # main, humaneval = the 164 HumanEval problems, bfcl = the 100
        # items of the bench's BFCL split. Declared so a coverage reading
        # has a denominator; a target not listed leaves it undeclared.
        "measure_size": {"aime2025": 30, "healthbench": 245, "arenahardwriting": 250,
                         "gpqamain": 448, "humaneval": 164, "bfcl": 100}.get(target),
        "task_apparatus": (f"post-training a small base LM for {target}; the "
                           "provided evaluate.py is graded material the agent "
                           "can run in-run on the public split; the bench's "
                           "held-out verdict is post-hoc and never enters the "
                           "record"),
    }


def _step(turn, scope, action, reasoning, obs, struct=None, tk=None,
          api_turn=None, session=0):
    s = {"turn": turn, "run": 0, "level": 1, "scope": scope,
         "action": action, "reasoning": reasoning,
         "testing": None, "expected": None, "surprised": False,
         "rejected": None, "obs": obs, "tool_kind": tk,
         "api_turn": api_turn, "session": session}
    if struct is not None:
        s["action_struct"] = struct
    return s


def _json_lines(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        ln = TS_PREFIX.sub("", ln.strip())
        if ln.startswith("{"):
            out.append(ln)
    return out


def _loads(lines):
    for ln in lines:
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            continue


def sniff_cursor(lines) -> bool:
    """cursor-cli stream: tool calls ride `tool_call` events (started /
    completed) whose body is one `<name>ToolCall` object; no other
    dialect here has that event type."""
    for ln in lines[:400]:
        try:
            if json.loads(ln).get("type") == "tool_call":
                return True
        except json.JSONDecodeError:
            continue
    return False


def _cursor_result(name: str, result: dict) -> str:
    """What the cursor harness returned for one call, as text: the
    payload field of each tool's `success` body (the dialect table), a
    failure body verbatim."""
    if not isinstance(result, dict) or not result:
        return ""
    ok = result.get("success")
    if not isinstance(ok, dict):
        return _txt(result)                     # failure / other: verbatim
    if name == "shellToolCall":
        out = ok.get("interleavedOutput") or "\n".join(
            x for x in (ok.get("stdout"), ok.get("stderr")) if x)
        tail = f"[exit={ok.get('exitCode')}]" if "exitCode" in ok else ""
        if result.get("isBackground"):
            tail = (tail + " " if tail else "") + "[background]"
        return (out + ("\n" if out and tail else "") + tail).strip()
    if name == "readToolCall":
        return str(ok.get("content") or "")
    if name == "globToolCall":
        return "\n".join(map(str, ok.get("files") or []))
    if name == "webSearchToolCall":
        return "\n".join(str(r.get("chunk") or "")
                         for r in ok.get("references") or []
                         if isinstance(r, dict))
    if name == "updateTodosToolCall":
        return "\n".join(("[x] " if "COMPLETED" in str(t.get("status")) else "[ ] ")
                         + str(t.get("content") or "") for t in ok.get("todos") or []
                         if isinstance(t, dict))
    if name == "editToolCall":
        return "\n".join(x for x in (str(ok.get("message") or ""),
                                     str(ok.get("diffString") or "")) if x)
    return _txt(ok)                              # await / grep: the body


def sniff_opencode(lines) -> bool:
    for ln in lines[:50]:
        try:
            t = json.loads(ln).get("type")
        except json.JSONDecodeError:
            continue
        if t in ("step_start", "step_finish"):
            return True
        if t in ("thread.started", "system", "assistant"):
            return False
    return False


def _drop_replays(steps: list) -> tuple[list, int]:
    """Drop transcript replays (base_adapter.replay_steps): a step is
    identified by its (action, obs, reasoning) triple, byte for byte."""
    keys = [(s.get("action"), s.get("obs"), s.get("reasoning")) for s in steps]
    drop = _base.replay_steps(keys, lambda i: bool(keys[i][0]))
    if not drop:
        return steps, 0
    kept = [s for i, s in enumerate(steps) if i not in drop]
    nxt = min((i for i in range(len(steps)) if i not in drop
               and i > min(drop)), default=None)
    if nxt is not None:
        s = steps[nxt]
        s["reasoning"] = (f"[{len(drop)} replayed steps dropped before this "
                          f"step: the harness re-emitted history]\n"
                          + (s.get("reasoning") or ""))
    return kept, len(drop)


class _Usage:
    """Per-api-turn token account (usage.json, grain api_call), the same
    rule as the speedrun converter: exact reasoning when the dump stamps
    it; else hidden thinking = per-turn residual out - visible_chars /
    CPT; the non-reasoning remainder char-split into text / action."""

    def __init__(self):
        self.per: dict[str, dict] = {}

    def row(self, t) -> dict:
        return self.per.setdefault(str(t), {"in": 0, "cached": 0, "out": 0,
                                           "reasoning": 0, "prose_chars": 0,
                                           "tool_chars": 0, "_stamped": False})

    def stamp(self, t, inp, cached, out, reasoning):
        r = self.row(t)
        if r["_stamped"]:          # one message id, several stream lines
            return
        r["_stamped"] = True
        r["in"] += int(inp or 0)
        r["cached"] += int(cached or 0)
        r["out"] += int(out or 0)
        r["reasoning"] += int(reasoning or 0)

    def prose(self, t, n):
        self.row(t)["prose_chars"] += n

    def tool(self, t, n):
        self.row(t)["tool_chars"] += n

    def emit(self, source: str) -> dict | None:
        if not any(r["out"] for r in self.per.values()):
            return None
        residual = not any(r["reasoning"] for r in self.per.values())
        turns = {}
        for t, r in self.per.items():
            if not r["out"]:
                continue
            tot = r["prose_chars"] + r["tool_chars"]
            reasoning = r["reasoning"]
            if residual:
                reasoning = int(round(max(0.0, r["out"] - tot / CPT)))
            nonr = max(0, r["out"] - reasoning)
            text = (nonr * r["prose_chars"] / tot) if tot else float(nonr)
            turns[t] = {"in": r["in"], "cached": r["cached"], "out": r["out"],
                        "thinking_tokens": reasoning,
                        "text_tokens": round(text),
                        "action_tokens": round(nonr - text),
                        "prose_chars": r["prose_chars"],
                        "tool_chars": r["tool_chars"]}
        return {"grain": "api_call",
                "reporter": (source + "; "
                             + ("no reasoning stamps: thinking = per-turn "
                                f"residual out - visible_chars/{CPT} (the "
                                "run-total band rule at its calibrated "
                                "end); " if residual else
                                "exact reasoning tokens as thinking_tokens; ")
                             + "non-reasoning remainder char-split into "
                               "text/action"),
                "turns": turns}


def _usage_armtotal(out_tok, reasoning_tok, prose, tool, source) -> dict:
    """One run-total usage line, no per-call stamps (as the speedrun
    converter's _emit_usage_armtotal). The run's reported out_tok is exact; the
    visible channel (assistant prose + agent-authored commands/patches)
    is countable; thinking is either EXACT (codex reasoning_output_tokens)
    or the RESIDUAL out - visible/cpt (claude-code runs whose backend
    reports no tokens per message and redacts thinking), banded by the
    chars-per-token ratio CPT_BAND, 2.9 (calibrated) .. 4.2 (generic) —
    both ends reported; T1's arm_budget_doing renders only when both
    ends agree on the pole."""
    exact = reasoning_tok is not None
    band = {}
    inconsistent = None
    for cpt in CPT_BAND:
        vis = (prose + tool) / cpt
        row = {"visible_tokens": round(vis),
               "think_side_share": round((out_tok - tool / cpt) / out_tok, 2)
               if out_tok else None,
               "do_side_share": round(tool / cpt / out_tok, 2) if out_tok else None}
        if exact:
            row["thinking_tokens_exact"] = reasoning_tok
        else:
            row["thinking_tokens_residual"] = round(out_tok - vis)
            if out_tok - vis < 0:
                # the visible channel alone exceeds the reported output:
                # the backend under-reports. No band is emitted — T1's arm_budget_doing abstains
                # rather than read a negative thinking budget.
                inconsistent = (f"visible_chars/{cpt} = {round(vis)} exceeds "
                                f"reported out_tok {out_tok}: the backend's "
                                "token account is not credible")
        band[str(cpt)] = row
    if inconsistent:
        band = {}
    doc = {"grain": "arm_total",
            "reporter": (source + "; run-total out_tok "
                         + ("and reasoning tokens exact" if exact else
                            "hard, thinking = residual out - visible_chars/cpt "
                            "(redacted, no per-call stamps)")
                         + "; visible channel char-counted and banded "
                           "chars/token 2.9 (calibrated) .. 4.2 (generic) for "
                           "the think/do split"),
            "out_tok": out_tok, "reasoning_tok": reasoning_tok,
            "prose_chars": prose, "tool_chars": tool, "band": band}
    if inconsistent:
        doc["inconsistent"] = inconsistent
    return doc


def build_steps(run: Path):
    lines = _json_lines((run / "solve_out.txt")
                        .read_text(errors="replace"))
    steps, delib = [], []
    pending: dict = {}          # claude tool_use id -> step index
    turn = 1
    scope_by = {"tool_kind": 0}
    usage = _Usage()
    usage_doc = None
    cur_api = None              # current api_turn (per-call dialects)
    session = 0                 # claude: one per closing result line
    worker_notes: dict = {}     # claude tool_use_id -> [progress lines]
    worker_step: dict = {}      # claude tool_use_id -> step index
    task_of: dict = {}          # claude task_id -> tool_use_id
    worker_task: dict = {}      # claude parent_tool_use_id -> task text
    worker_begun: set = set()   # workers whose first step is marked
    worker_seen: set = set()    # workers with any inlined event
    gaps = {"file_changes": 0, "file_changes_without_payload": 0,
            "web_search_results_absent": 0,
            "calls_without_result": 0,
            "subagent_calls_without_transcript": 0,
            "subagent_transcripts_inlined": 0,
            "replay_steps_dropped": 0}
    # harness events that are not the agent's acts but shape the record:
    # a context compaction (the agent lost its working memory), a rate
    # limit or API retry (the process stalled), a tool error (the result
    # the agent read was an error), thinking excluded by policy
    events = {"context_compactions": 0, "rate_limit_events": 0,
              "api_retries": 0, "tool_errors": 0,
              "thinking_chars_excluded_by_policy": 0}
    task_prompt: str | None = None
    task_desc: dict = {}        # dispatch call id -> its one-line description
    pending_compact: str | None = None

    def flush_delib():
        nonlocal delib
        s = "\n".join(delib).strip()
        delib = []
        return s

    def kind_of(name, cmd: str) -> str:
        """The kind of tool the harness ran — a fact from its name (the
        shared table raises on a name it does not list)."""
        tk = tool_kind(name, cmd)
        scope_by["tool_kind"] += 1
        return tk

    def push(tk: str, action: str, obs: str, cmd: str = "",
             patches=None, delegation=None, actor=None):
        nonlocal turn
        scope = "env" if tk in ENV_KINDS else "workspace"
        if tk == "poll" and not cmd.strip():
            # a poll submits no command; the action names the poll so
            # the judges see what the step was (the base adapter renders "(poll)")
            action = f"$ (poll) {action.lstrip('$ ').strip()}".rstrip()
        struct = _exec_struct(cmd, "readlog" if tk == "poll" else "exec") \
            if scope == "env" else None
        st = _step(turn, scope, action, flush_delib(), obs, struct, tk,
                   cur_api, session)
        # the struct is a harness fact (kind / program / files named);
        # the observation channel carries ONLY what the world returned;
        # the full command already sits on the ACTION line.
        if patches:
            st["patches"] = patches
            gaps["file_changes"] += 1
            if not any(str(p.get("new") or "").strip()
                       or str(p.get("old") or "").strip() for p in patches):
                gaps["file_changes_without_payload"] += 1
        if delegation:
            st["delegation"] = delegation
        if actor:
            # a worker's own step, inlined at its true position: the
            # first one carries the task it was handed (as the speedrun
            # converter marks it)
            st["actor"] = actor
            wid = actor.split(":", 1)[1]
            if wid not in worker_begun:
                worker_begun.add(wid)
                # the dispatch call's one-line description when known
                # (the full instruction rides that step's action), else
                # the instruction itself, in full
                task = (task_desc.get(wid) or worker_task.get(wid) or "").strip()
                head = f"[worker {wid} begins; task: {task}]"
            else:
                head = f"[worker {wid}]"
            st["reasoning"] = head + ("\n" + st["reasoning"] if st["reasoning"] else "")
        steps.append(st)
        if scope == "env":
            turn += 1
        return st

    def push_harness(text: str, note: str = "", actor=None):
        """A message the agent RECEIVED — its own step, the text in the
        observation channel, kept apart from the agent's words: a compaction handoff summary, the task
        prompt, a nudge. It does not consume the agent's pending
        deliberation: that still rides the agent's own next action."""
        nonlocal delib
        saved, delib = delib, []
        st = push("message", "(harness message)", text, actor=actor)
        if note:
            st["reasoning"] = (st["reasoning"] + "\n" + note) if st["reasoning"] else note
        if actor is None:
            st["actor"] = "harness"
        delib = saved
        return st

    def close_open_calls():
        """A harness prompt while claude calls are open: those calls
        never returned — the dump holds no answer."""
        for tid, i in list(pending.items()):
            steps[i]["obs"] = NO_RESULT
            gaps["calls_without_result"] += 1
            pending.pop(tid, None)

    # ------------------------------------------------------------ codex
    if sniff_codex(lines):
        dialect = "codex"
        prose_chars = tool_chars = 0
        run_usage = None
        started: dict = {}          # command id -> command (item.started)
        for o in _loads(lines):
            ot = o.get("type")
            if ot == "turn.completed" and isinstance(o.get("usage"), dict):
                run_usage = o["usage"]
                continue
            if ot == "item.started":
                it = o.get("item") or {}
                if it.get("type") == "command_execution":
                    started[it.get("id")] = _unwrap_codex(it.get("command") or "")
                continue
            if ot != "item.completed":
                continue
            it = o.get("item") or {}
            t = it.get("type")
            if t == "reasoning":
                events["thinking_chars_excluded_by_policy"] += len(it.get("text") or "")
                continue
            if t == "agent_message":
                txt = (it.get("text") or "").strip()
                if txt:
                    delib.append(txt)
                    prose_chars += len(txt)
            elif t == "web_search":
                # a read of the web; the dump carries the query only —
                # the results the agent saw are not in it
                q = str(it.get("query") or (it.get("action") or {}).get("query") or "")
                scope_by["tool_kind"] += 1
                tool_chars += len(q)
                # the observation channel carries only what the harness
                # returned: the item's status. Results are not in the
                # dump — counted, not narrated.
                gaps["web_search_results_absent"] += 1
                push("read", f"web_search {q}", str(it.get("status") or ""))
            elif t == "todo_list":
                items = it.get("items") or []
                done = sum(1 for x in items if isinstance(x, dict) and x.get("completed"))
                body = "\n".join(("[x] " if x.get("completed") else "[ ] ") + str(x.get("text") or "")
                                 for x in items if isinstance(x, dict))
                scope_by["tool_kind"] += 1
                tool_chars += len(body)
                push("bookkeeping", f"todo_list {len(items)} items, {done} done\n{body}", "")
            elif t == "command_execution":
                started.pop(it.get("id"), None)
                # command_execution IS a shell call; an empty command is
                # the codex idle exec that only reads what the running
                # process printed since (poll) — the base adapter's rule
                cmd = _unwrap_codex(it.get("command") or "")
                tk = "shell" if cmd.strip() else "poll"
                scope_by["tool_kind"] += 1
                tool_chars += len(cmd)
                obs = _txt(it.get("aggregated_output"))
                obs += f"\n[exit={it.get('exit_code')}]"
                push(tk, f"$ {_txt(cmd)}", obs, cmd)
            elif t == "file_change":
                # codex records WHICH files changed, never the content:
                # {path, kind} only — an edit without payload (counted)
                scope_by["tool_kind"] += 1
                chs = it.get("changes") or []
                patches = [{"path": str(c.get("path") or ""), "new": "",
                            "kind": str(c.get("kind") or "")}
                           for c in chs if isinstance(c, dict)] or \
                          [{"path": str(it.get("path") or ""), "new": ""}]
                files = " ".join(p["path"] for p in patches)
                push("edit", f"file_change {files}",
                     str(it.get("status") or ""), patches=patches)
            elif t == "mcp_tool_call":
                name = it.get("tool") or it.get("name") or "mcp"
                args = _txt(it.get("arguments") or {})
                tk = kind_of(name, args)
                tool_chars += len(args)
                out = it.get("result") or it.get("output")
                push(tk, f"{name} {args}", result_text(out, None), args)
        # a command the harness started and never completed: the call
        # was made, the dump holds no answer
        for cid, cmd in started.items():
            scope_by["tool_kind"] += 1
            push("shell", f"$ {_txt(cmd)}", NO_RESULT, cmd)
            gaps["calls_without_result"] += 1
        if run_usage:
            usage_doc = _usage_armtotal(
                int(run_usage.get("output_tokens") or 0),
                int(run_usage.get("reasoning_output_tokens") or 0),
                prose_chars, tool_chars,
                "codex event stream turn.completed.usage")
    # ------------------------------------------------------- cursor-cli
    elif sniff_cursor(lines):
        # cursor-cli: user /
        # assistant messages as JSON, thinking as deltas (excluded by
        # policy), every tool call as a started + completed pair whose
        # body is one <name>ToolCall {args, result}; the names are in the
        # shared table (an unlisted one stops the conversion); one usage
        # block on the closing result line (arm_total, thinking redacted)
        dialect = "cursor-cli"
        prose_chars = tool_chars = 0
        run_usage = None
        open_calls: dict = {}          # toolCallId -> step index
        for o in _loads(lines):
            t, sub = o.get("type"), o.get("subtype")
            if t == "user":
                text = "\n".join(str(b.get("text") or "") for b in
                                 ((o.get("message") or {}).get("content") or [])
                                 if isinstance(b, dict) and b.get("type") == "text")
                if text.strip():
                    if task_prompt is None:
                        task_prompt = text
                    push_harness(text)
            elif t == "assistant":
                for b in (o.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "text" \
                            and str(b.get("text") or "").strip():
                        delib.append(str(b["text"]))
                        prose_chars += len(b["text"])
            elif t == "thinking":
                if sub == "delta":
                    events["thinking_chars_excluded_by_policy"] += \
                        len(o.get("text") or "")
            elif t == "tool_call":
                tc = o.get("tool_call") or {}
                name = next((k for k in tc if str(k).endswith("ToolCall")), None)
                if name is None:
                    raise SystemExit("cursor-cli: tool_call without a "
                                     f"<name>ToolCall body: {str(tc)[:160]!r}")
                body = tc.get(name) or {}
                args = body.get("args") or {}
                cid = tc.get("toolCallId") or o.get("call_id")
                if sub == "started":
                    cmd = str(args.get("command") or "")
                    tk = kind_of(name, cmd or _txt(args))
                    tool_chars += len(cmd) if cmd else len(_txt(args))
                    desc = str(args.get("description")
                               or body.get("description") or "").strip()
                    if desc and tk in ENV_KINDS:
                        delib.append(desc)          # the agent's words at the call
                    if tk in ENV_KINDS:
                        # a poll names its call and args (taskId, the regex
                        # awaited on); push() marks the empty command "(poll)"
                        action = (f"$ {name} {_txt(args)}"
                                  if not cmd.strip() else f"$ {_txt(cmd)}")
                        push(tk, action, "", cmd)
                    else:
                        patches = [{"path": str(args.get("path") or ""),
                                    "new": str(args.get("streamContent") or "")}] \
                            if tk == "edit" else None
                        push(tk, f"{name} {_txt(args)}", "", patches=patches)
                    open_calls[cid] = len(steps) - 1
                elif sub == "completed":
                    i = open_calls.pop(cid, None)
                    if i is None:
                        continue                    # a completion with no start
                    res = body.get("result") or {}
                    text = _cursor_result(name, res)
                    if isinstance(res, dict) and "success" not in res:
                        events["tool_errors"] += 1
                        text = "[tool error]\n" + text
                    steps[i]["obs"] = text
            elif t == "system" and sub == "task_notification":
                push_harness(f"status={o.get('status')}\n{o.get('title') or ''}",
                             note=f"[notification: background task {o.get('task_id')} "
                                  f"{o.get('status') or 'ended'}]")
            elif t == "retry" and sub == "starting":
                events["api_retries"] += 1
            elif t == "result":
                if isinstance(o.get("usage"), dict):
                    run_usage = o["usage"]
        for cid, i in open_calls.items():       # started, never completed
            steps[i]["obs"] = NO_RESULT
            gaps["calls_without_result"] += 1
        usage_doc = None
        if run_usage:
            usage_doc = _usage_armtotal(
                int(run_usage.get("outputTokens") or 0), None,
                prose_chars, tool_chars,
                "cursor-cli closing result line usage.outputTokens for the whole "
                "run (no per-call stamps; thinking redacted -> residual)")
    # --------------------------------------------------------- opencode
    elif sniff_opencode(lines):
        dialect = "opencode"
        mid_order: dict = {}

        def api_of(mid):
            if mid is None:
                return cur_api
            if mid not in mid_order:
                mid_order[mid] = len(mid_order) + 1
            return mid_order[mid]
        for o in _loads(lines):
            t = o.get("type")
            p = o.get("part") or {}
            if t == "text":
                cur_api = api_of(p.get("messageID"))
                txt = (p.get("text") or "").strip()
                if txt:
                    delib.append(txt)
                    usage.prose(cur_api, len(txt))
            elif t == "step_finish":
                cur_api = api_of(p.get("messageID"))
                tk_ = p.get("tokens") or {}
                cache = tk_.get("cache") or {}
                usage.stamp(cur_api, tk_.get("input"),
                            (cache.get("read") or 0) + (cache.get("write") or 0),
                            tk_.get("output"), tk_.get("reasoning"))
            elif t == "tool_use":
                cur_api = api_of(p.get("messageID"))
                state = p.get("state") or {}
                tool, inp = p.get("tool") or "?", state.get("input") or {}
                cmd = str(inp.get("command") or "")
                tk = kind_of(tool, cmd or _txt(inp))
                usage.tool(cur_api, len(cmd) if cmd else len(_txt(inp)))
                out = state.get("output")
                # the call's one-line description is the agent's own
                # words at that step (as for claude-code); env calls
                # render only the command, so keep it
                desc = str(inp.get("description") or "").strip() \
                    if isinstance(inp, dict) else ""
                if desc and tk in ENV_KINDS:
                    delib.append(desc)
                if tk in ENV_KINDS:
                    push(tk, f"$ {_txt(cmd)}" if cmd.strip()
                         else f"{tool} {_txt(inp)}",
                         _txt(out), cmd)
                else:
                    push(tk, f"{tool} {_txt(inp)}",
                         _txt(out),
                         patches=_edit_patches(tool, inp) if tk == "edit" else None)
        usage_doc = usage.emit("opencode step_finish.part.tokens per message "
                               "(input/output/reasoning/cache)")
    # ------------------------------------------------------ claude-code
    else:
        dialect = "claude-code"
        mid_order: dict = {}
        # A dump may hold SEVERAL sessions (the harness may re-invoke the
        # CLI for follow-ups); every result
        # line closes one session and carries that session's totals, so
        # the run total is their SUM, and each step carries its session.
        run_total: dict | None = None
        for o in _loads(lines):
            t = o.get("type")
            # a worker's (subagent's) event, inlined in the parent stream
            # at its true position: parent_tool_use_id names the Task /
            # Agent call that dispatched it
            parent = o.get("parent_tool_use_id") or None
            if parent:
                worker_seen.add(parent)
            if t == "result" and isinstance(o.get("usage"), dict):
                u_ = o["usage"]
                mu = o.get("modelUsage") or {}
                if run_total is None:
                    run_total = {"out_tok": 0, "in_tok": 0, "cache_read": 0,
                                 "models_seen": {}, "num_turns": 0,
                                 "total_cost_usd": 0.0, "n_result_lines": 0}
                run_total["out_tok"] += int(u_.get("output_tokens") or 0)
                run_total["in_tok"] += int(u_.get("input_tokens") or 0)
                run_total["cache_read"] += int(u_.get("cache_read_input_tokens") or 0)
                run_total["num_turns"] += int(o.get("num_turns") or 0)
                run_total["total_cost_usd"] += float(o.get("total_cost_usd") or 0)
                run_total["n_result_lines"] += 1
                # usage.* on a result line is that SESSION's account
                # (summed above); modelUsage is CUMULATIVE since process
                # start (single-session dumps: equal to usage) — the
                # last result line carries the run's per-model total
                if isinstance(mu, dict):
                    run_total["models_seen"] = {
                        k: int((v or {}).get("outputTokens") or 0)
                        for k, v in mu.items()}
                session += 1
                continue
            if t == "assistant":
                if pending_compact:
                    push_harness("", note=pending_compact)
                    pending_compact = None
                m = o.get("message") or {}
                mid = m.get("id")
                if mid is not None:
                    if mid not in mid_order:
                        mid_order[mid] = len(mid_order) + 1
                    cur_api = mid_order[mid]
                u = m.get("usage") or {}
                if u:
                    usage.stamp(cur_api, u.get("input_tokens"),
                                (u.get("cache_read_input_tokens") or 0)
                                + (u.get("cache_creation_input_tokens") or 0),
                                u.get("output_tokens"),
                                u.get("reasoning_tokens"))   # absent: redacted
                for b in m.get("content") or []:
                    bt = b.get("type")
                    if bt == "text" and (b.get("text") or "").strip():
                        delib.append(b["text"])
                        usage.prose(cur_api, len(b["text"]))
                    elif bt == "thinking":
                        events["thinking_chars_excluded_by_policy"] += len(
                            b.get("thinking") or "")
                    elif bt == "tool_use":
                        name, inp = b.get("name") or "?", b.get("input") or {}
                        cmd = str(inp.get("command") or "")
                        tk = kind_of(name, cmd or _txt(inp))
                        usage.tool(cur_api, len(cmd) if cmd else len(_txt(inp)))
                        deleg = "dispatch+relay" if tk == "delegate" else None
                        actor = f"worker:{parent}" if parent else None
                        # the one-line description the agent writes on a
                        # call ("Check GPU memory") is its own statement
                        # of what the call is for — its words at that
                        # step, kept.
                        # Workspace calls already render their whole input.
                        desc = str(inp.get("description") or "").strip() \
                            if isinstance(inp, dict) else ""
                        if desc and tk in ENV_KINDS:
                            delib.append(desc)
                        if deleg and desc:
                            task_desc[b.get("id")] = desc
                        if tk in ENV_KINDS:
                            head_ = ("monitor" if name == "Monitor" else "$")
                            st = push(tk, f"{head_} {_txt(cmd)}" if cmd.strip()
                                      else f"{name} {_txt(inp)}", "", cmd,
                                      actor=actor)
                        else:
                            st = push(tk, f"{name} {_txt(inp)}", "",
                                      patches=_edit_patches(name, inp)
                                      if tk == "edit" else None,
                                      delegation=deleg, actor=actor)
                        pending[b.get("id")] = len(steps) - 1
                        if deleg:
                            worker_step[b.get("id")] = len(steps) - 1
            elif t == "user":
                content = (o.get("message") or {}).get("content")
                texts = []
                if isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text" and str(b.get("text") or "").strip():
                            texts.append(str(b["text"]))
                            continue
                        if b.get("type") != "tool_result":
                            continue
                        i = pending.pop(b.get("tool_use_id"), None)
                        if i is None:
                            continue
                        cc = b.get("content")
                        text = result_text(cc, None)
                        # a result block that only references a tool
                        # (TaskCreate / TaskUpdate results) has no text;
                        # say so instead of leaving the obs blank
                        if isinstance(cc, list):
                            refs = [x.get("tool_name") for x in cc
                                    if isinstance(x, dict) and x.get("type") == "tool_reference"]
                            if refs:
                                text = (text + "\n" if text else "") + \
                                    "[tool_reference: " + ", ".join(map(str, refs)) + "]"
                        if b.get("is_error"):
                            events["tool_errors"] += 1
                            text = "[tool error]\n" + text
                        if b.get("tool_use_id") in worker_step:
                            # the worker's WRITTEN report — a same-model
                            # transcription, not first-hand world output
                            text = (f"[relay <- worker "
                                    f"{b.get('tool_use_id')}: written report]\n"
                                    + text)
                        prefix = steps[i]["obs"]
                        steps[i]["obs"] = (prefix + text) if prefix else text
                elif isinstance(content, str) and content.strip():
                    texts.append(content)
                if texts:
                    # a user-role text in this dialect is one of three
                    # things: the task handed to a worker (parent_tool_use_id
                    # set), the summary the agent was given after a
                    # context compaction, or the run's own task prompt.
                    # Never a turn cut: calls pair by id, exactly.
                    text = "\n".join(texts)
                    if parent and parent not in worker_task:
                        # a worker's FIRST message is the parent's
                        # instruction: it rides the dispatch step's
                        # action and the worker's opening marker
                        worker_task[parent] = text
                    elif parent:
                        push_harness(text, actor=f"worker:{parent}")
                    elif text.startswith("This session is being continued"):
                        push_harness(text, note=pending_compact or "")
                        pending_compact = None
                    else:
                        if task_prompt is None:
                            task_prompt = text
                        push_harness(text)
            elif t == "rate_limit_event":
                events["rate_limit_events"] += 1
            elif t == "system":
                sub = o.get("subtype")
                if sub == "thinking_tokens":
                    # the CLI's per-block thinking ESTIMATE (delta), the
                    # only per-turn thinking signal this dialect carries
                    usage.row(cur_api)["reasoning"] += int(
                        o.get("estimated_tokens_delta") or 0)
                elif sub == "compact_boundary":
                    # the agent's context was compacted: it lost its
                    # working memory here — a fact the judges must see
                    events["context_compactions"] += 1
                    meta_ = o.get("compact_metadata") or {}
                    pending_compact = (f"[context compacted ({meta_.get('trigger')}) "
                                       f"at ~{meta_.get('pre_tokens')} tokens]")
                elif sub == "api_retry":
                    events["api_retries"] += 1
                elif sub in ("task_started", "task_progress", "task_notification",
                             "task_completed", "task_updated"):
                    # worker lifecycle lines: task_started carries both
                    # ids; task_updated only task_id — map it back
                    tid = o.get("tool_use_id")
                    task_id = o.get("task_id")
                    if tid and task_id:
                        task_of[task_id] = tid
                    tid = tid or task_of.get(task_id)
                    if sub == "task_notification":
                        # the completion notice the agent RECEIVED, with
                        # the worker's summary: a message, its own step
                        body = str(o.get("summary") or o.get("description")
                                   or o.get("message") or "")
                        if o.get("status"):
                            body = f"status={o['status']}\n{body}"
                        # the notice names a worker only when the call it
                        # closes was a dispatch; otherwise it is a
                        # background command the agent launched (the
                        # summary is the call's own description)
                        if tid in worker_step:
                            note = f"[relay <- worker {tid}: completion notice]"
                        else:
                            note = (f"[notification: background task {task_id} "
                                    f"of call {tid} {o.get('status') or 'ended'}]")
                        push_harness(body, note=note)
                        continue
                    if sub == "task_updated":
                        desc = _txt(o.get("patch") or {})[:200]
                    else:
                        desc = str(o.get("description") or o.get("summary")
                                   or o.get("prompt") or "")[:300]
                        if o.get("status"):
                            desc = f"status={o['status']}; {desc}"
                    worker_notes.setdefault(tid, []).append(f"{sub}: {desc}")
        close_open_calls()                 # calls the dump never answered
        # a Task/Agent call whose worker left no inlined event has no
        # transcript in the dump; the rest are inlined and actor-marked
        gaps["subagent_calls_without_transcript"] = sum(
            1 for tid in worker_step if tid not in worker_seen)
        gaps["subagent_transcripts_inlined"] = sum(
            1 for tid in worker_step if tid in worker_seen)
        # the harness's own account of each worker (task_started /
        # progress / notification / updated lines) rides the dispatch
        # step's reasoning channel
        task_name = {v: k for k, v in task_of.items()}
        for tid, notes in worker_notes.items():
            i = worker_step.get(tid)
            if i is None:
                continue
            head = (f"[worker {task_name.get(tid, tid)} progress, per the "
                    f"harness: " + " | ".join(notes)[:1500] + "]")
            steps[i]["reasoning"] = head + ("\n" + steps[i]["reasoning"]
                                            if steps[i]["reasoning"] else "")
        # claude-code carries NO usable per-call account: a message's usage
        # stamp is the snapshot at its first content block — output_tokens
        # 3..7, constant across the message's lines — and thinking is
        # redacted, so the per-message sum is a small fraction of the
        # run's output. Emitting api_call from those stamps would
        # feed T1's per-belief windows numbers that are not the truth.
        # The stream's closing result line is the one exact account:
        # arm_total, thinking = residual (as the speedrun converter does).
        prose_sum = sum(r["prose_chars"] for r in usage.per.values())
        tool_sum = sum(r["tool_chars"] for r in usage.per.values())
        per_msg_out = sum(r["out"] for r in usage.per.values())
        usage_doc = None
        if run_total and run_total["out_tok"]:
            usage_doc = _usage_armtotal(
                run_total["out_tok"], None, prose_sum, tool_sum,
                "claude stream-json closing result line (usage.output_tokens "
                "for the whole run; per-message usage stamps are first-block "
                "snapshots, not a ledger — rejected)")
            usage_doc["run_total"] = run_total
            usage_doc["run_total"]["per_message_out_sum_rejected"] = per_msg_out
            usage_doc["run_total"]["source"] = (
                f"sum of {run_total['n_result_lines']} CLI result line(s)' "
                "usage (one per session)")
            # the CLI's own per-block thinking ESTIMATES (system /
            # thinking_tokens deltas, chipbench's reading of this dialect)
            # ride beside the residual as an independent second reading;
            # not rescaled — the result line carries no exact thinking
            # total in this bench (no output_tokens_details)
            by_turn = {t: r["reasoning"] for t, r in usage.per.items()
                       if r["reasoning"]}
            est = sum(by_turn.values())
            resid = (usage_doc.get("band") or {}).get("2.9", {}).get(
                "thinking_tokens_residual")
            usage_doc["thinking_estimate"] = {
                "source": ("estimate (CLI thinking_tokens deltas, unrescaled: "
                           "the result line carries no exact thinking total)"),
                "total": est,
                "api_turns_with_estimate": len(by_turn),
                "by_api_turn": by_turn,
                "residual_at_2.9_for_comparison": resid,
                "estimate_over_residual": (round(est / resid, 2)
                                           if resid and resid > 0 else None)}

    steps, dropped = _drop_replays(steps)
    gaps["replay_steps_dropped"] = dropped
    meta = {"scope_by": scope_by, "record_gaps": gaps, "usage": usage_doc,
            "events": events, "task_prompt": task_prompt}
    return steps, delib, dialect, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = Path(a.out_dir)
    steps, delib_tail, dialect, meta = build_steps(run)
    if not steps:
        # an empty dump is a failed download, not a silent run: the HF
        # pull can store the server's "Entry not found" body as
        # solve_out.txt, and a 0-step record would ride the bench as a
        # unit that did nothing — refuse instead.
        head = (run / "solve_out.txt").read_text(errors="replace")[:80] \
            if (run / "solve_out.txt").exists() else "(no solve_out.txt)"
        raise SystemExit(f"{run.name}: dump yields no steps (dialect "
                         f"{dialect}; head: {head!r}) — refusing to write "
                         f"an empty record")
    out.mkdir(parents=True, exist_ok=True)

    source = (run / "SOURCE.txt").read_text().strip() \
        if (run / "SOURCE.txt").exists() else run.name
    # SOURCE.txt is either the bare "<agent>/<unit>" path or a descriptive
    # line that embeds it ("aisa-group/PostTrainBench-Trajectories ::
    # <agent>/<unit> (pulled ...)"); the unit ends in "_<run id digits>".
    # Take the embedded path, not a split of the whole line on "/".
    m = re.search(r"([\w.\-]+)/([\w.\-]+_\d+)(?=\s|$|/|\))", source)
    if m:
        agent_dir, exp = m.group(1), m.group(2)
    else:
        agent_dir, exp = (source.split("/") + [""])[:2]
    world = re.sub(r"_\d+$", "", exp) or exp
    target = world.split("_")[0] if world else "?"

    # The task as the agent read it. Most dumps do not carry the opening
    # goal message (it went in on the command line); then the bench's own
    # template, filled for this run, stands in — a bench-side fact from the
    # bench's code (task_prompt.py), and the manifest says which it was.
    if meta["task_prompt"]:
        task_prompt_text, task_prompt_source = meta["task_prompt"], "dump (opening goal message)"
    else:
        task_prompt_text, note = task_prompt.reconstruct(run, f"{agent_dir}/{exp}")
        if task_prompt_text is None:
            task_prompt_text, task_prompt_source = "(goal message absent from dump)", f"absent: {note}"
        else:
            task_prompt_source = note

    sub = "\n".join(delib_tail).strip()
    # the closing step carries the agent's last message and the one
    # harness fact that the transcript ended; the budget cut and the
    # post-hoc grading are the world declaration (world.json), not
    # something the agent observed (a run may also finish early).
    steps.append(_step(max((s["turn"] for s in steps), default=0),
                       "submission", "(transcript ends)", sub, "",
                       tk="message",
                       api_turn=steps[-1].get("api_turn") if steps else None))
    for i, s in enumerate(steps):
        s["gid"] = i

    n_env = sum(1 for s in steps if s["scope"] == "env")
    kinds: dict = {}
    for s in steps:
        kinds[s["tool_kind"]] = kinds.get(s["tool_kind"], 0) + 1
    gaps = meta["record_gaps"]
    gaps["note"] = ("a file change with no payload rides the record as an "
                    "action head only — the extractor sees THAT a file "
                    "changed, not what changed (codex records only "
                    "{path, kind}); a subagent call rides as a delegate "
                    "step whose obs is the worker's written report — the "
                    "worker's own transcript is not in the dump")
    usage_doc = meta["usage"]
    thinking = {
        "claude-code": ("redacted in the stream; the per-message usage stamps "
                        "are first-block snapshots (output_tokens 3..7), not a "
                        "ledger, so usage.json is the run total from the "
                        "closing result line (arm_total) with thinking = "
                        f"residual out - visible_chars/{CPT}"),
        "codex": ("absent from the stream; reasoning_output_tokens exact "
                  "for the whole run in usage.json (arm_total)"),
        "opencode": ("absent from the stream; per-message reasoning tokens "
                     "exact in usage.json"),
        "cursor-cli": ("streamed as thinking deltas, excluded by policy "
                       "(chars counted in harness_events); usage.json is the "
                       "closing result line's run total (arm_total) with "
                       f"thinking = residual out - visible_chars/{CPT}"),
    }[dialect]
    manifest = {
        "bench": "posttrainbench",
        "world": world,
        "model": agent_dir,
        "source_trajectory": source,
        "n_env_steps": n_env,
        "n_workspace_steps": sum(1 for s in steps
                                 if s["scope"] == "workspace"),
        "scope_by": meta["scope_by"],
        "steps_by_tool_kind": kinds,
        "record_gaps": gaps,
        # harness events that shaped the record but are not the agent's
        # acts (see build_steps): compactions ride the reasoning channel
        # as markers; the rest are counted here
        "harness_events": meta["events"],
        "task_prompt": task_prompt_text,
        "task_prompt_source": task_prompt_source,
        "usage_grain": (usage_doc or {}).get("grain") if usage_doc else "none",
        "scope_law": ("scope is the KIND of tool the harness ran (tool_kind on "
                      "every step): shell calls and polls of a running process "
                      "observe the container (env); reads, edits, delegations "
                      "and bookkeeping do not (workspace). Never a reading of "
                      "the command text; what a step showed or launched is the "
                      "extractor's reading of its action and observation"),
        "scope_rulings": {
            "shell call (claude Bash / opencode bash / codex "
            "command_execution), whatever the command":
                "env exec — the container ran something; whether it was a "
                "training launch, an eval, or a `sed -n` of a script is the "
                "extractor's reading, not this adapter's",
            "poll of a running process (TaskOutput, BashOutput, Monitor; "
            "codex empty command)":
                "env readlog — reads what a live run printed since",
            "file reads/edits/writes, delegations, task-list bookkeeping":
                "workspace — authoring and housekeeping observe nothing; "
                "edits carry patches (old -> new) when the harness recorded "
                "them",
            "claude Task/Agent (a subagent)":
                "workspace delegate, delegation=dispatch+relay: the call "
                "dispatched a worker, its result relays the worker's written "
                "report (same-model prose); harness task_started/progress "
                "lines ride the step's reasoning",
            "final message at time-out": "submission",
            "bench-side eval (metrics.json/judge_output.json)":
                "EXCLUDED — outcome oracle never enters the record",
        },
        "harness": dialect,
        "channel_policy": "text+tools (unified tier: thinking excluded "
                          "by policy)",
        "channels": {
            "official_reply": "absent (nothing replies to the agent "
                              "during the run)",
            "transcript_text": "visible",
            "transcript_thinking": thinking,
        },
        "textualizer": "benches/posttrainbench/adapter.py: tool_kind "
                       "is the scope fact, no text fallback (unknown tool "
                       "name stops); patches on edits; api_turn + usage.json; "
                       "delegation on subagent calls; NO_RESULT on unanswered "
                       "calls; replay drop; record_gaps; world.json declared "
                       "from code",
    }
    (out / "record.jsonl").write_text("\n".join(json.dumps(s,
                                                ensure_ascii=False)
                                                for s in steps))
    (out / "record_manifest.json").write_text(json.dumps(manifest, indent=1))
    # the world declaration, written from code like the base, speedrun
    # and chipbench adapters (never by hand); the module docstring says
    # which parts the scorer reads
    (out / "world.json").write_text(json.dumps(world_declaration(target),
                                               indent=1))
    up = out / "usage.json"
    if usage_doc:
        up.write_text(json.dumps(usage_doc, indent=1))
    elif up.exists():
        up.unlink()
    sb = meta["scope_by"]
    print(f"{run.name}: {len(steps)} steps ({n_env} env / "
          f"{manifest['n_workspace_steps']} ws / 1 submission, {dialect}; "
          f"scope by tool_kind {sb['tool_kind']}; patches {gaps['file_changes']} "
          f"({gaps['file_changes_without_payload']} without payload); "
          f"usage {manifest['usage_grain']}) -> {out}")


if __name__ == "__main__":
    main()
