"""Single LLM entry point for every oeb tool: hardened transport and a replayable call journal.

Journal ("ledger"): every successful model call — request and response, full
text — is appended to <unit>/calls.jsonl. Re-running the same build hits the
journal and replays byte-identically at zero cost; with OEB_LLM_REPLAY_ONLY
set, a call the journal cannot answer stops the run instead of reaching the
model. The journal is the audit trail: any verdict in the graph can be
traced to the judge's verbatim answer here.

Wiring: ONE journal per unit, <unit>/calls.jsonl, shared by every stage —
each stage's main() opens it. The request hash keys every answer, so
stages never collide in one file. A tool with no unit dir journals when
OEB_LLM_LEDGER=<path/to/calls.jsonl> is set; with neither, calls are NOT journaled (transport hardening still
applies).

Within one process identical requests are memoized: two calls with the same
(model, prompt, salt) return ONE sample. Sampling loops that need
independent draws MUST pass distinct salts (vote index, extraction-run
index) — the jury ballots and the extraction passes do.

Routing by model name: glm* -> GLM endpoint (HTTP, or the claude CLI with
GLM_TRANSPORT=cli); gpt* -> codex CLI (HTTP with GPT_TRANSPORT=http);
kimi* -> kimi CLI; grok* -> grok CLI; gemini* -> Antigravity CLI (agy);
deepseek* -> dsh CLI (HTTP with DEEPSEEK_TRANSPORT=http); anything else is
a Claude model on the transport below.

Transport (default "cli"): `claude -p` hardened against environment
contamination — fixed --system-prompt (marker OEB-JUDGE-v1), empty sandbox
cwd under the system temp dir (no CLAUDE.md auto-discovery from any repo),
hooks disabled, MCP config stripped, tools denied, no session persistence.
Residual: the CLI's own scaffolding, pinned by recording `claude --version`
in each journal's header. OEB_LLM_TRANSPORT=api switches to the raw
Anthropic SDK (needs ANTHROPIC_API_KEY; byte-exact prompt control); cli
is the default.

Retry: transient failures back off 0/30/90/240s — an outage must degrade
to a pause, not kill an hours-long pipeline. Failed calls are never
journaled; a refusal is journaled as a refusal, so replay reproduces it.

Self-test:  python -m oeb.llm selftest [--model claude-sonnet-5]
verifies the isolation empirically (system prompt in effect, no ambient
context, no tools).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# repo-root .env: provider keys, loaded at import as DEFAULTS ONLY — a
# variable already in the environment always wins. Values never
# printed, never passed on argv. When this package is vendored as a
# submodule, the .env may sit one level up, so try both roots.
def _load_dotenv() -> None:
    here = Path(__file__).resolve()
    cand = [here.parents[1] / ".env"]        # package root (standalone use)
    if len(here.parents) > 2:
        cand.append(here.parents[2] / ".env")   # host repo root
    p = next((c for c in cand if c.is_file()), cand[0])
    try:
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(),
                                  v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

SYSTEM = (
    "You are OEB-JUDGE-v1, a scoring and annotation function inside a "
    "research pipeline. Follow the task in the user message exactly and "
    "output ONLY what it asks for (usually bare JSON) — no preamble, no "
    "markdown fences, no commentary. Base every answer solely on the text "
    "provided in the user message; you have no tools, no filesystem access, "
    "and no outside context."
)
RETRY_DELAYS = (0, 30, 90, 240)
RETRYABLE_HTTP = (408, 429, 500, 502, 503, 504, 529)


class _Transient(Exception):
    """One failed transport attempt; the message becomes the final
    RuntimeError once every retry is spent."""


def _retry_loop(attempt):
    """The transport retry skeleton: backoff per RETRY_DELAYS;
    attempt(i) returns the reply or raises _Transient(msg) to burn
    one try — any other exception (Refused, auth failures, hard HTTP
    codes) propagates untouched."""
    last = ""
    for i, d in enumerate(RETRY_DELAYS):
        if d:
            time.sleep(d)
        try:
            return attempt(i)
        except _Transient as e:
            last = str(e)
    raise RuntimeError(last)
_DENY_TOOLS = ["*"]      # wildcard: judge gets NO tools, any CLI version

# Provider-side safeguard refusals are a property of the PROMPT (the record's
# own content), not of the moment: retrying is guaranteed waste, and the
# refusal must reach the caller as a MISSING measurement, never as an empty
# answer that downstream parsers would read as a real zero.
REFUSAL_MARKERS = (
    "has safety measures that flagged this message",
    "Cyber Verification Program",
    # CLI prints this to STDOUT with rc=1 and EMPTY stderr — without a
    # marker it reads as a blank transient error and burns all retries
    "appears to violate our Usage Policy",
    # OpenAI HTTP 400 invalid_request_error wording: deterministic
    # prompt-content flag, same family as the CLI markers above
    "flagged as potentially violating our usage policy",
)


class Refused(RuntimeError):
    """The provider declined to answer this prompt (deterministic)."""

    def __init__(self, model: str, detail: str):
        super().__init__(f"refused by {model}: {detail[:200]}")
        self.model, self.detail = model, detail

_lock = threading.Lock()
_memo: dict[str, str] = {}                 # h -> response
_inflight: dict[str, threading.Event] = {}
_errs: dict[str, BaseException] = {}
_refused: dict[str, str] = {}              # h -> refusal detail (no answer)
_ledger_path: Path | None = None
_ledger_fh = None
_sandbox: str | None = None
_cli_ver: str | None = None
_gate_sem: threading.Semaphore | None = None


def _gate() -> threading.Semaphore:
    """The fresh-call concurrency gate: one process-wide cap on
    transports actually in flight, whatever the nesting of caller
    fan-outs (jury pools, extraction passes, run()'s jury pool all
    stack). Memo hits, ledger replays and in-flight
    waits never touch it. Width: OEB_LLM_CONCURRENCY, default 6 (the
    same as the default JURY_WORKERS)."""
    global _gate_sem
    with _lock:
        if _gate_sem is None:
            _gate_sem = threading.BoundedSemaphore(
                int(os.environ.get("OEB_LLM_CONCURRENCY", "6")))
        return _gate_sem


def sha(text: str | bytes, n: int = 12) -> str:
    b = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(b).hexdigest()[:n]


def transport() -> str:
    return os.environ.get("OEB_LLM_TRANSPORT", "cli")


def cli_version() -> str:
    global _cli_ver
    if _cli_ver is None:
        try:
            _cli_ver = subprocess.run(["claude", "--version"],
                                      capture_output=True, text=True,
                                      timeout=30).stdout.strip()
        except Exception:
            _cli_ver = "unknown"
    return _cli_ver


def transport_desc() -> str:
    if transport() == "api":
        try:
            import anthropic
            return f"api anthropic-sdk/{anthropic.__version__}"
        except Exception:
            return "api anthropic-sdk/unknown"
    return f"cli {cli_version()}"


def _req_hash(model: str, prompt: str, salt: str) -> str:
    env = {"v": 1, "t": transport(), "m": model, "sys": sha(SYSTEM, 16),
           "s": salt, "p": prompt}
    return hashlib.sha256(json.dumps(env, sort_keys=True,
                                     ensure_ascii=True).encode()).hexdigest()


def _load_entries(path: Path) -> tuple[dict | None, dict[str, str]]:
    """Read a journal: (header or None, {h: response}).

    Refusal records carry no response — a refused prompt stays refused, so
    they reload into _refused and are replayed as refusals, never as
    answers.
    """
    header, entries = None, {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            d = json.loads(line)
            if i == 0 and "_ledger" in d:
                header = d
            elif d.get("refused") and "h" in d:
                _refused[d["h"]] = d.get("detail", "")
            elif "h" in d and "response" in d:
                entries[d["h"]] = d["response"]
    return header, entries


def init(ledger: str | Path) -> None:
    """Open (or create) this run's journal. Re-entrant: resets state."""
    global _ledger_path, _ledger_fh
    with _lock:
        if _ledger_fh:
            _ledger_fh.close()
        _memo.clear(), _errs.clear(), _refused.clear()
        _ledger_path = Path(ledger)
        _ledger_path.parent.mkdir(parents=True, exist_ok=True)
        header = None
        if _ledger_path.exists():
            header, entries = _load_entries(_ledger_path)
            _memo.update(entries)
        if header is None:
            header = {"_ledger": 1,
                      "run_id": f"r{time.strftime('%Y%m%d-%H%M%S')}"
                                f"-{os.getpid()}",
                      "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "transport": transport_desc(), "system": SYSTEM}
            with open(_ledger_path, "a") as fh:
                fh.write(json.dumps(header, ensure_ascii=True) + "\n")
        _ledger_fh = open(_ledger_path, "a")


def _append(rec: dict) -> None:
    if _ledger_fh:
        _ledger_fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
        _ledger_fh.flush()


def _sandbox_dir() -> str:
    """Empty cwd for the CLI: nothing to read, no CLAUDE.md in any
    ancestor (system temp), so project/memory auto-discovery finds nothing."""
    global _sandbox
    if _sandbox is None:
        _sandbox = tempfile.mkdtemp(prefix="oeb-llm-sandbox-")
    return _sandbox


def _cli_cmd(model: str) -> list[str]:
    return ["claude", "-p", "--model", model,
            "--system-prompt", SYSTEM,
            "--settings", '{"disableAllHooks": true}',
            "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
            "--no-session-persistence",
            "--disallowedTools", *_DENY_TOOLS]


def _call_cli(model: str, prompt: str, timeout: int,
              env: dict | None = None, extra: tuple[str, ...] = ()) -> str:
    def attempt(i):
        try:
            r = subprocess.run(_cli_cmd(model) + list(extra), input=prompt,
                               capture_output=True, text=True,
                               timeout=timeout, cwd=_sandbox_dir(),
                               env=env)
        except subprocess.TimeoutExpired:   # outage windows time out too
            raise _Transient(
                f"llm timeout {timeout}s (try {i + 1}/{len(RETRY_DELAYS)})")
        if r.returncode == 0:
            return r.stdout
        blob = (r.stdout or "") + (r.stderr or "")
        if any(m in blob for m in REFUSAL_MARKERS):
            raise Refused(model, blob.strip())     # content-determined: no retry
        raise _Transient(
            f"llm rc={r.returncode} (try {i + 1}/{len(RETRY_DELAYS)}): "
            f"{r.stderr[-300:]}")
    return _retry_loop(attempt)


def _gpt_effort() -> str:
    """Reasoning effort for gpt* calls through the codex CLI
    (OEB_GPT_EFFORT, default medium). Pinned because the CLI would
    otherwise inherit ~/.codex/config.toml, a per-machine setting that
    must not leak into the instrument. The HTTP path sends no effort
    field (gateways disagree on its name), so the API default, also
    medium, applies."""
    return os.environ.get("OEB_GPT_EFFORT", "medium")


# codex ships as a coding agent: every exec carries its own agent prompt,
# skill catalog, plugin/app/multi-agent notes and tool schemas — 17,914
# input tokens before the judge prompt (measured on a "Reply OK" probe).
# The judge is a pure text call, so all of it is switched off: 4,573.
_CODEX_OFF_FEATURES = (
    "apps", "browser_use", "browser_use_external", "computer_use", "goals",
    "image_generation", "multi_agent", "personality", "plugins",
    "remote_plugin", "skill_search", "skill_mcp_dependency_install",
    "tool_suggest", "view_image", "sleep_tool", "hooks",
    "workspace_dependencies", "shell_tool", "unified_exec", "shell_snapshot")
_CODEX_OFF_CONFIG = (
    'web_search="disabled"', "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_permissions_instructions=false",
    "include_environment_context=false")
_CODEX_INSTRUCTIONS = ("You are a text-only function. Answer the request "
                       "directly in plain text. Do not use tools.\n")
_codex_instr: str | None = None


def _codex_trim_args() -> list[str]:
    """Flags that strip codex's agent scaffolding. Its built-in agent
    prompt is replaced by a one-line file (model_instructions_file) kept
    outside the judge sandbox, so the sandbox stays empty."""
    global _codex_instr
    with _lock:
        if _codex_instr is None:
            fd, p = tempfile.mkstemp(prefix="oeb-codex-instr-", suffix=".md")
            with os.fdopen(fd, "w") as fh:
                fh.write(_CODEX_INSTRUCTIONS)
            _codex_instr = p
    args = []
    for f in _CODEX_OFF_FEATURES:
        args += ["--disable", f]
    for c in _CODEX_OFF_CONFIG:
        args += ["-c", c]
    return args + ["-c", f'model_instructions_file="{_codex_instr}"']


def _call_codex_cli(model: str, prompt: str, timeout: int) -> str:
    """OpenAI models through the installed codex CLI: one-shot `codex
    exec` in the same empty sandbox dir the claude CLI uses — read-only,
    ephemeral, no repo, no user rules, agent scaffolding stripped
    (_codex_trim_args). codex has no --system-prompt, so SYSTEM rides at the top of the prompt. The
    judge is a pure text call; the sandbox holds only the per-call
    last-message file."""
    def attempt(i):
        fd, outp = tempfile.mkstemp(suffix=".txt", dir=_sandbox_dir())
        os.close(fd)
        try:
            r = subprocess.run(
                ["codex", "exec", "-", "-m", model, "-s", "read-only",
                 "-c", "model_reasoning_effort=" + _gpt_effort(),
                 *_codex_trim_args(),
                 "--skip-git-repo-check", "--ephemeral", "--color",
                 "never", "-C", _sandbox_dir(), "-o", outp],
                input=SYSTEM + "\n\n" + prompt, capture_output=True,
                text=True, timeout=timeout, cwd=_sandbox_dir())
        except subprocess.TimeoutExpired:
            raise _Transient(
                f"codex timeout {timeout}s (try {i + 1}/{len(RETRY_DELAYS)})")
        finally:
            msg = Path(outp).read_text() if os.path.exists(outp) else ""
            os.path.exists(outp) and os.unlink(outp)
        if r.returncode == 0 and msg.strip():
            return msg
        blob = (r.stdout or "") + (r.stderr or "")
        if any(m in blob for m in REFUSAL_MARKERS):
            raise Refused(model, blob.strip())
        raise _Transient(
            f"codex rc={r.returncode} empty={not msg.strip()} "
            f"(try {i + 1}/{len(RETRY_DELAYS)}): {blob[-300:]}")
    return _retry_loop(attempt)


def _call_kimi_cli(model: str, prompt: str, timeout: int) -> str:
    """kimi models through the installed kimi-code CLI: one-shot
    `kimi -p` in the shared empty sandbox. No --system-prompt flag, so
    SYSTEM rides the prompt.
    Model: 'kimi-k3' routes to the CLI's default alias (kimi-for-coding,
    the k3 line); any other kimi-* name is passed through as -m alias.
    Requires a live `kimi login` OAuth session."""
    def attempt(i):
        cmd = ["kimi", "-p", SYSTEM + "\n\n" + prompt,
               "--output-format", "stream-json"]   # lossless: plain text
        if model != "kimi-k3":                 # mode bullet-prefixes lines
            cmd[1:1] = ["-m", model]           # k3 = config default alias
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=_sandbox_dir())
        except subprocess.TimeoutExpired:
            raise _Transient(
                f"kimi timeout {timeout}s (try {i + 1}/{len(RETRY_DELAYS)})")
        if r.returncode == 0 and (r.stdout or "").strip():
            msg = ""
            for line in r.stdout.splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("role") == "assistant" and ev.get("content"):
                    msg = ev["content"]
            if msg.strip():
                return msg
            raise _Transient(f"kimi rc=0 but no assistant event "
                             f"(try {i + 1}/{len(RETRY_DELAYS)})")
        blob = (r.stdout or "") + (r.stderr or "")
        if "login_required" in blob:
            raise RuntimeError("kimi CLI needs `kimi login` (OAuth) "
                               "before judge calls can run")
        if any(m in blob for m in REFUSAL_MARKERS):
            raise Refused(model, blob.strip())
        raise _Transient(
            f"kimi rc={r.returncode} empty={not (r.stdout or '').strip()} "
            f"(try {i + 1}/{len(RETRY_DELAYS)}): {blob[-300:]}")
    return _retry_loop(attempt)


# grok CLI ("Grok Build", xAI; OAuth session under ~/.grok). It is a
# coding agent: every headless call ships its toolset (28 tools, 14.4k
# input tokens for a one-line prompt);
# an agent-profile file does not strip them, and `--tools ""` is ignored,
# but a denylist naming every built-in does (4.0k for the same prompt,
# and a "read canary.txt if you have any tool" probe answers NO_TOOLS).
# The list is the union the agent self-reported across probes plus the
# documented names; an unknown name is ignored by the CLI.
_GROK_DENY_TOOLS = (
    "x_user_search,x_semantic_search,x_keyword_search,x_thread_fetch,"
    "run_terminal_command,run_terminal_cmd,read_file,search_replace,"
    "list_dir,grep,kill_command_or_subagent,todo_write,"
    "get_command_or_subagent_output,wait_commands_or_subagents,"
    "scheduler_create,scheduler_delete,scheduler_list,monitor,"
    "search_tool,use_tool,workflow,enter_plan_mode,exit_plan_mode,"
    "ask_user_question,send_feedback,image_gen,image_edit,"
    "image_to_video,reference_to_video,write,web_search,web_fetch,Agent")


def _call_grok_cli(model: str, prompt: str, timeout: int) -> str:
    """grok models through the installed grok CLI. One-shot headless
    call in the shared empty sandbox: the prompt rides a temp file
    (--prompt-file; argv would E2BIG on 150k-char windows), --verbatim so @file / slash expansion never touches it,
    SYSTEM as the system-prompt override, every built-in tool denied,
    no subagents / web / plan, one turn. --output-format json; the
    reply is the object's `text` (reasoning stays in `thought`, not
    read). Effort pinned by OEB_GROK_EFFORT (default medium, the gpt
    judge's level). Credentials stay in ~/.grok (OAuth); nothing on
    argv but the prompt path."""
    effort = os.environ.get("OEB_GROK_EFFORT", "medium")
    env = dict(os.environ, GROK_DISABLE_AUTOUPDATER="1",
               GROK_TELEMETRY_TRACE_UPLOAD="0", GROK_SUBAGENTS="0",
               GROK_WEB_FETCH="0", GROK_BACKEND_SEARCH="0")
    fd, pf = tempfile.mkstemp(prefix="grok-prompt-", suffix=".txt",
                              dir=_sandbox_dir())
    with os.fdopen(fd, "w") as fh:
        fh.write(prompt)
    cmd = ["grok", "--prompt-file", pf, "--verbatim", "-m", model,
           "--effort", effort, "--system-prompt-override", SYSTEM,
           "--disallowed-tools", _GROK_DENY_TOOLS, "--no-subagents",
           "--disable-web-search", "--no-plan", "--max-turns", "1",
           "--permission-mode", "dontAsk", "--output-format", "json"]

    def attempt(i):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=_sandbox_dir(), env=env)
        except subprocess.TimeoutExpired:
            raise _Transient(
                f"grok timeout {timeout}s (try {i + 1}/{len(RETRY_DELAYS)})")
        blob = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and (r.stdout or "").strip():
            try:
                out = json.loads(r.stdout)
            except json.JSONDecodeError:
                raise _Transient(f"grok rc=0 but stdout is not the json "
                                 f"object (try {i + 1}): {r.stdout[-300:]}")
            msg = out.get("text") or ""
            if msg.strip():
                return msg
            raise _Transient(f"grok empty text (try {i + 1}/"
                             f"{len(RETRY_DELAYS)}): stop={out.get('stopReason')}")
        if any(m in blob for m in REFUSAL_MARKERS):
            raise Refused(model, blob.strip())
        if "login" in blob.lower() and ("expired" in blob.lower()
                                       or "unauthorized" in blob.lower()
                                       or "not logged" in blob.lower()):
            raise RuntimeError(f"grok CLI needs `grok login`: {blob.strip()[-200:]}")
        raise _Transient(
            f"grok rc={r.returncode} empty={not (r.stdout or '').strip()} "
            f"(try {i + 1}/{len(RETRY_DELAYS)}): {blob[-300:]}")
    try:
        return _retry_loop(attempt)
    finally:
        try:
            os.unlink(pf)
        except OSError:
            pass

_AGY_AGENT = "oeb-judge"


def _agy_agent_dir() -> str:
    """The CLI's global agent that makes a judge call a plain text call.
    The Antigravity CLI is a coding agent: by default every
    call carries its agent prompt and 57 tool schemas (~13.9k input tokens
    on a one-line prompt, measured) and the model may act on them — on
    the extractor prompt gemini-3.8-flash ran tools and answered "{}".
    A custom agent with excludeDefaultComponents: true (agy >= 1.2.1)
    drops all of that: 592 input tokens on the same one-line prompt, no
    tools in the session. Discovery is the global config directory
    (~/.gemini/config/agents/<name>/agent.md — workspace .agents/ was not
    found from a print-mode cwd); the file is (re)written on every
    process start so the definition can never drift from this code. The
    judge SYSTEM text is the agent body, so the prompt rides alone."""
    d = os.path.join(os.path.expanduser("~"), ".gemini", "config", "agents", _AGY_AGENT)
    os.makedirs(d, exist_ok=True)
    body = ("---\nname: " + _AGY_AGENT + "\ndescription: Plain-text judge for oeb "
            "instruments. Answers the prompt directly; no tools, no workspace, no "
            "default prompt sections.\nexcludeDefaultComponents: true\n---\n"
            + SYSTEM + "\n")
    p = os.path.join(d, "agent.md")
    try:
        cur = open(p).read()
    except OSError:
        cur = None
    if cur != body:
        with open(p, "w") as fh:
            fh.write(body)
    global _agy_checked
    if not _agy_checked:
        # once per process: the CLI silently falls back to its default
        # agent when the named one does not resolve (log-only warning),
        # and its init event lists the tool catalog either way — the one
        # visible difference is the input side of a one-line call
        # (~13.9k tokens default, ~600 as the judge agent).
        r = subprocess.run(["agy", "--agent", _AGY_AGENT, "-p",
                            "Reply with exactly the word OK and nothing else.",
                            "--model", "gemini-3.8-flash-low", "--output-format", "json",
                            "--sandbox", "--disable-slash-commands"],
                           capture_output=True, text=True, timeout=180, cwd=_sandbox_dir())
        try:
            usage = json.loads((r.stdout or "")[(r.stdout or "").find("{"):])["usage"]
        except (ValueError, KeyError, TypeError):
            raise RuntimeError(f"agy probe call failed: {(r.stdout or '')[-300:]} {(r.stderr or '')[-300:]}")
        if usage.get("input_tokens", 10 ** 9) > 3000:
            raise RuntimeError(
                f"agy did not apply the {_AGY_AGENT} agent ({usage.get('input_tokens')} "
                f"input tokens on a one-line probe; ~600 expected) — check {p}")
        _agy_checked = True
    return d


_agy_checked = False


def _call_agy_cli(model: str, prompt: str, timeout: int) -> str:
    """Gemini models through the Antigravity CLI (agy), as the
    _AGY_AGENT plain-text agent (see _agy_agent_dir). The prompt rides
    stdin as one stream-json user event — a judge prompt is longer than one argv string may be (a
    150k-char -p fails with "Argument list too long") — and the reply
    is the stream's final `result` event. Effort is part of the model
    id (gemini-3.8-flash-high). --sandbox restricts the terminal."""
    _agy_agent_dir()
    def attempt(i):
        cmd = ["agy", "--agent", _AGY_AGENT, "-p", "", "--model", model,
               "--input-format", "stream-json", "--output-format", "stream-json",
               "--sandbox", "--disable-slash-commands",
               "--print-timeout", f"{timeout}s"]
        stdin = json.dumps({"event": "user", "message": {
            "role": "user", "content": prompt}}) + "\n"
        try:
            r = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                               timeout=timeout + 30, cwd=_sandbox_dir())
        except subprocess.TimeoutExpired:
            raise _Transient(
                f"agy timeout {timeout}s (try {i + 1}/{len(RETRY_DELAYS)})")
        result = None
        for line in (r.stdout or "").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "result":
                result = ev.get("result") or {}
        if result and result.get("status") == "SUCCESS" and (result.get("response") or "").strip():
            return result["response"]
        blob = (r.stdout or "")[-1500:] + (r.stderr or "")[-500:]
        if any(m in blob for m in REFUSAL_MARKERS):
            raise Refused(model, blob.strip())
        raise _Transient(
            f"agy rc={r.returncode} status={(result or {}).get('status')} "
            f"(try {i + 1}/{len(RETRY_DELAYS)}): {((result or {}).get('error') or blob)[-300:]}")
    return _retry_loop(attempt)


def _call_glm_cli(model: str, prompt: str, timeout: int) -> str:
    """GLM models through the claude CLI pointed at the Anthropic-
    compatible GLM endpoint (the same transport as the Claude judge).
    Credentials enter through the
    subprocess environment only — never argv, never logs; the CLI's
    connector warning lands on stderr, stdout stays the pure reply.

    The CLI is a coding agent; the judge is a pure text call, so
    everything else is switched off: --bare
    (no hooks, plugin sync, memory, CLAUDE.md discovery), no tool
    schemas, no skills, and no nonessential traffic — without that last
    one every call sends a second background request to the GLM quota.
    Probe: one request of 183 input tokens for a one-line prompt.
    Effort is pinned (OEB_GLM_EFFORT, default medium = the gpt judge's
    level) so a per-machine setting cannot leak into the instrument."""
    env = dict(os.environ,
               ANTHROPIC_BASE_URL=os.environ["GLM_BASE_URL"],
               ANTHROPIC_AUTH_TOKEN=os.environ["GLM_API_KEY"],
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    extra = ("--bare", "--tools", "", "--disable-slash-commands",
             "--effort", os.environ.get("OEB_GLM_EFFORT", "medium"))
    return _call_cli(model, prompt, timeout, env=env, extra=extra)


def _call_glm_http(model: str, prompt: str, timeout: int) -> str:
    """Anthropic-compatible third-party endpoint over stdlib HTTP — no
    SDK dependency. Credentials come from the environment, never from
    code or logs."""
    import urllib.request
    import urllib.error
    url = os.environ["GLM_BASE_URL"].rstrip("/") + "/v1/messages"
    body = json.dumps({
        "model": model, "max_tokens": 8192, "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}]}).encode()
    def attempt(i):
        req = urllib.request.Request(url, data=body, headers={
            "content-type": "application/json",
            "x-api-key": os.environ["GLM_API_KEY"],
            "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            return "".join(b.get("text", "") for b in
                           out.get("content", [])
                           if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = f"glm http {e.code} (try {i + 1}): {detail}"
            if e.code not in RETRYABLE_HTTP:
                raise RuntimeError(last)
            raise _Transient(last)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise _Transient(f"glm connection (try {i + 1}): {e}")
    return _retry_loop(attempt)


def _dsh_patch_path(model: str) -> str:
    """Compose the dsh judge profile patch for one model and cache it in
    the shared sandbox. The patch pins the judge persona as the harness
    system prompt, disables the action tools (subagents, editor, web),
    and declares the provider route: an sk-or-* DEEPSEEK_API_KEY rides
    OpenRouter (model id kept provider-prefixed), anything else rides
    the official endpoint (prefix stripped, key read natively)."""
    openrouter = os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-or-")
    if openrouter:
        mid = model if "/" in model else "deepseek/" + model
        route = ("- id: agent-default-model\n"
                 "  config:\n"
                 "    provider: openrouter\n"
                 f"    model: {json.dumps(mid)}\n"
                 "- id: llm-pi-ai\n"
                 "  config:\n"
                 "    providers:\n"
                 "      openrouter:\n"
                 "        apiKeyEnv: OPENROUTER_API_KEY\n"
                 "        displayName: OpenRouter\n"
                 "        api: openai-completions\n"
                 "        baseURL: https://openrouter.ai/api/v1\n"
                 "        models:\n"
                 f"          - id: {json.dumps(mid)}\n"
                 f"            name: {json.dumps(mid)}\n")
    else:
        mid = model.split("/", 1)[-1]
        route = ("- id: agent-default-model\n"
                 "  config:\n"
                 "    provider: deepseek-official\n"
                 f"    model: {json.dumps(mid)}\n")
    # every action tool off (verified: the tools self-report probe
    # returns [] under this list); the headless system prompt is not
    # overridable via the persona config, so SYSTEM rides at the top of
    # the task text instead — the codex-judge pattern.
    off = ("tool-bash tool-pwsh tool-jobs tool-fs tool-fs-search "
           "tool-skill skill skill-filesystem skill-badge "
           "tool-subagent-control tool-subagent-list-agents tool-subagent "
           "tool-subagent-fork tool-subagent-report tool-workflow "
           "tool-todo tool-goal goal goal-round-driver command-goal "
           "plan-mode user-questions tool-ralph tool-str-replace-editor "
           "tool-web web-search-deepseek").split()
    patch = route + "".join(f"- id: {t}\n  disabled: true\n" for t in off)
    p = os.path.join(_sandbox_dir(),
                     f"dsh-judge-{hashlib.md5(patch.encode()).hexdigest()[:10]}.yml")
    if not os.path.exists(p):
        Path(p).write_text(patch)
    return p


def _call_dsh_cli(model: str, prompt: str, timeout: int) -> str:
    """DeepSeek models through the installed dsh harness CLI (the
    vendor's own harness, same pattern as codex/kimi). One-shot
    `dsh --profile headless` in the shared empty sandbox; the judge
    persona and tool disables ride a profile patch (dsh has no
    --system-prompt); credentials enter through the
    subprocess environment only — never argv, never logs. DSH_HOME is
    kept inside the sandbox so no user profile or session state leaks
    into judge calls."""
    # the task text rides argv; Linux caps a single argv string at
    # MAX_ARG_STRLEN (~128KB), so oversized windows (giant apply_patch
    # actions, late-run rosters) delegate to the HTTP transport — same
    # model, same provider, SYSTEM as the system message (avoids E2BIG)
    # deepseek's reasoning tail routinely exceeds the default 600s cap
    # (median 300-500s/call) — transport-level floor, this judge only
    timeout = max(timeout, 1800)
    if len(SYSTEM) + len(prompt) > 100_000:
        return _call_deepseek_http(model, prompt, timeout)
    import shutil
    dsh = shutil.which("dsh") or os.path.expanduser(
        "~/.npm-global/bin/dsh")
    home = os.path.join(_sandbox_dir(), "dsh-home")
    os.makedirs(home, exist_ok=True)
    env = dict(os.environ, DSH_HOME=home, DSH_PERMISSION_MODE="read-only",
               DSH_TELEMETRY_MODE="DISABLED")
    if os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-or-"):
        env["OPENROUTER_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    patch = _dsh_patch_path(model)
    def attempt(i):
        try:
            r = subprocess.run(
                [dsh, "--profile", "headless", "--patch", patch,
                 SYSTEM + "\n\n" + prompt],
                capture_output=True, text=True, timeout=timeout,
                cwd=_sandbox_dir(), env=env)
        except subprocess.TimeoutExpired:
            raise _Transient(
                f"dsh timeout {timeout}s (try {i + 1}/{len(RETRY_DELAYS)})")
        if r.returncode == 0 and (r.stdout or "").strip():
            return r.stdout
        blob = (r.stdout or "") + (r.stderr or "")
        if any(m in blob for m in REFUSAL_MARKERS):
            raise Refused(model, blob.strip())
        if "AUTH" in blob and "invalid" in blob:
            raise RuntimeError(f"dsh auth: {blob.strip()[:200]}")
        raise _Transient(
            f"dsh rc={r.returncode} empty={not (r.stdout or '').strip()} "
            f"(try {i + 1}/{len(RETRY_DELAYS)}): {blob[-300:]}")
    return _retry_loop(attempt)


def _post_chat(url, auth_key, body, timeout, tag, retry_empty):
    """Shared OpenAI-compatible chat-completions POST: stdlib HTTP, RETRYABLE_HTTP
    status classification, IncompleteRead-class connection errors
    retried; retry_empty additionally burns a try on a blank reply
    (the openai rule — deepseek returns blanks as-is)."""
    import urllib.request
    import urllib.error

    def attempt(i):
        req = urllib.request.Request(url, data=body, headers={
            "content-type": "application/json",
            "authorization": "Bearer " + auth_key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            msg = (out.get("choices") or [{}])[0].get(
                "message", {}).get("content", "") or ""
            if retry_empty and not msg.strip():
                raise _Transient(f"{tag} empty reply (try {i + 1})")
            return msg
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = f"{tag} http {e.code} (try {i + 1}): {detail}"
            if any(m in detail for m in REFUSAL_MARKERS):
                raise Refused(tag, detail)  # content-determined: no retry
            if e.code not in RETRYABLE_HTTP:
                raise RuntimeError(last)
            raise _Transient(last)
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as e:
            # HTTPException covers IncompleteRead: the gateway can cut
            # the chunked body mid-stream on minutes-long generations
            # over oversized prompts — transient, retry
            raise _Transient(f"{tag} connection (try {i + 1}): "
                             f"{type(e).__name__} {e}")
    return _retry_loop(attempt)


def _call_deepseek_http(model: str, prompt: str, timeout: int) -> str:
    """OpenAI-compatible endpoint for deepseek-* judges over stdlib
    HTTP — no SDK dependency. Credentials come from the environment, never from code or logs.
    DEEPSEEK_BASE_URL defaults to the official API; point it at an
    OpenRouter-style gateway to use `deepseek/...` model paths — the
    provider prefix is stripped only for the official endpoint."""
    timeout = max(timeout, 1800)      # same reasoning-tail floor as
                                      # the dsh path
    default_base = ("https://openrouter.ai/api/v1"
                    if os.environ.get("DEEPSEEK_API_KEY",
                                      "").startswith("sk-or-")
                    else "https://api.deepseek.com")
    base = os.environ.get("DEEPSEEK_BASE_URL", default_base).rstrip("/")
    url = base + "/chat/completions"
    mid = model
    if "deepseek.com" in base and mid.startswith("deepseek/"):
        mid = mid.split("/", 1)[1]
    elif "openrouter" in base and "/" not in mid:
        mid = "deepseek/" + mid
    body = json.dumps({
        "model": mid, "max_tokens": 8192,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}]}).encode()
    return _post_chat(url, os.environ["DEEPSEEK_API_KEY"], body,
                      timeout, "deepseek", retry_empty=False)


def _call_openai_http(model: str, prompt: str, timeout: int) -> str:
    """OpenAI models over the HTTP API with OPENAI_API_KEY (env or
    repo-root .env). Opt-in: gpt* models go through the codex CLI
    unless GPT_TRANSPORT=http. OPENAI_BASE_URL overrides the endpoint (any
    chat-completions-compatible gateway); OPENAI_MODEL_<name with
    - -> _> remaps a model id when the gateway names it differently."""
    timeout = max(timeout, 1800)     # reasoning-tail floor (dsh rule)
    base = os.environ.get("OPENAI_BASE_URL",
                          "https://api.openai.com/v1").rstrip("/")
    url = base + "/chat/completions"
    mid = os.environ.get("OPENAI_MODEL_" + model.replace("-", "_"),
                         model)
    # this model family rejects max_tokens (400: use
    # max_completion_tokens). 16384 not 8192: the reasoning tail spends
    # from the same budget, and a hard window can burn all of it before
    # the answer starts — the reply then comes back EMPTY
    body = json.dumps({
        "model": mid, "max_completion_tokens": 16384,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}]}).encode()
    return _post_chat(url, os.environ["OPENAI_API_KEY"], body,
                      timeout, "openai", retry_empty=True)


def _call_api(model: str, prompt: str, timeout: int) -> str:
    import anthropic
    client = anthropic.Anthropic()
    def attempt(i):
        try:
            msg = client.messages.create(
                model=model, max_tokens=8192, system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout)
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text")
        except anthropic.APIStatusError as e:      # transient server side
            # 408 deliberately absent from the SDK path's retryable set
            if e.status_code not in (429, 500, 502, 503, 504, 529):
                raise
            raise _Transient(f"llm api {e.status_code} (try {i + 1}): {e}")
        except anthropic.APIConnectionError as e:  # incl. timeouts
            raise _Transient(f"llm api connection (try {i + 1}): {e}")
    return _retry_loop(attempt)


def call(model: str, prompt: str, timeout: int = 600, salt: str = "") -> str:
    """The one entry point. Identical (model, prompt, salt) => one sample."""
    if _ledger_path is None and os.environ.get("OEB_LLM_LEDGER"):
        init(os.environ["OEB_LLM_LEDGER"])
    h = _req_hash(model, prompt, salt)
    while True:
        with _lock:
            if h in _refused:            # a refused prompt stays refused
                raise Refused(model, _refused[h])
            if h in _memo:
                return _memo[h]
            ev = _inflight.get(h)
            if ev is None:
                _inflight[h] = threading.Event()
                break
        ev.wait()
        with _lock:
            if h in _errs:
                raise _errs[h]
        # else loop back and read the memo

    if os.environ.get("OEB_LLM_REPLAY_ONLY"):
        # a zero-cost re-read of a journal must never turn into a paid
        # call — a miss is a changed prompt, name it and stop.
        with _lock:
            _inflight.pop(h).set()
        raise RuntimeError(f"llm: OEB_LLM_REPLAY_ONLY set and the journal has no "
                           f"answer for this prompt (model={model} salt={salt!r} h={h[:12]})")

    t0 = time.time()
    try:
        if model.startswith("glm"):
            fn = (_call_glm_cli if os.environ.get("GLM_TRANSPORT") == "cli"
                  else _call_glm_http)
        elif model.startswith("gpt"):
            fn = (_call_openai_http
                  if os.environ.get("GPT_TRANSPORT") == "http"
                  else _call_codex_cli)
        elif model.startswith("kimi"):
            fn = _call_kimi_cli
        elif model.startswith("grok"):
            fn = _call_grok_cli
        elif model.startswith("gemini"):
            fn = _call_agy_cli
        elif model.startswith("deepseek"):
            fn = (_call_deepseek_http
                  if os.environ.get("DEEPSEEK_TRANSPORT") == "http"
                  else _call_dsh_cli)
        else:
            fn = _call_api if transport() == "api" else _call_cli
        with _gate():
            resp = fn(model, prompt, timeout)
    except Refused as e:
        with _lock:
            _refused[h] = e.detail
            _errs[h] = e
            _append({"h": h, "model": model, "salt": salt,
                     "prompt": prompt, "refused": True, "detail": e.detail})
            _inflight.pop(h).set()
        raise
    except BaseException as e:
        with _lock:
            _errs[h] = e
            _inflight.pop(h).set()
        raise
    with _lock:
        _memo[h] = resp
        _append({"h": h, "model": model, "salt": salt,
                 "ms": int(1000 * (time.time() - t0)),
                 "prompt": prompt, "response": resp})
        _inflight.pop(h).set()
    return resp


# ------------------------------------------------------------------ selftest

_PROBES = [
    ("system prompt in effect",
     "State the exact marker token from your system prompt, and nothing "
     "else.",
     lambda r: "OEB-JUDGE-v1" in r),
    ("no ambient context leaked",
     "List the names of any project files, CLAUDE.md instructions, memory "
     "files, MCP servers, or skills visible in your context. If there are "
     "none, reply exactly NONE.",
     lambda r: "NONE" in r.upper() and "CLAUDE.md" not in r),
    ("no tools available",
     "List the names of every tool you can invoke right now. If you have "
     "no tools, reply exactly NO-TOOLS.",
     lambda r: "NO-TOOLS" in r.upper().replace(" ", "-")),
]


def selftest(model: str) -> int:
    print(f"transport: {transport_desc()}")
    print(f"cmd: {' '.join(_cli_cmd(model))}  cwd={_sandbox_dir()}")
    bad = 0
    for name, probe, ok in _PROBES:
        try:
            r = call(model, probe, timeout=120, salt="selftest").strip()
        except Exception as e:
            print(f"FAIL  {name}: call error {e}")
            bad += 1
            continue
        verdict = "PASS" if ok(r) else "WARN"
        bad += verdict == "WARN"
        print(f"{verdict}  {name}: {r[:200]!r}")
    print("note: probes are model-self-reports — a WARN means inspect the "
          "raw answer, not necessarily a leak.")
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest", help="verify transport isolation")
    st.add_argument("--model", default="claude-sonnet-5")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(1 if selftest(a.model) else 0)
