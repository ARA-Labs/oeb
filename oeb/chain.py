"""The scoring chain: one command takes a unit from record to readings.

    python -m oeb.chain <unit_dir> [<unit_dir> ...] --model gpt-5.6-sol
                        [--jobs 4] [--retries 12] [--wait 300]

Stages, in order, each its own `python -m` process:

    extract -> connect -> assemble -> uptake -> overreach
    -> analogy -> reward_hacking -> coverage -> measure

Extraction passes and jury votes are the stages' own defaults (3 and 3);
the chain has no knob for them.

Every judge answer lands in <unit>/calls.jsonl (oeb.llm), so a failed
attempt starts again from the first stage and all that was already
answered replays at no cost. A failed attempt waits --wait seconds first:
a quota or network outage must turn into a pause, not a dead batch.

<unit>/chain.log receives every stage's output, each attempt headed by the
commit that ran it ("+dirty" when the checkout had uncommitted edits — a
process runs the files on disk, not a commit).

--jobs runs that many units side by side; each stage process also opens
up to OEB_LLM_CONCURRENCY judge calls at once, so the judge sees
jobs x OEB_LLM_CONCURRENCY in flight.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent


def stages(model: str) -> list[tuple[str, list[str]]]:
    """The stages in order, minus any named in OEB_CHAIN_SKIP.

    OEB_CHAIN_SKIP is a comma-separated list of stage names (e.g. "coverage").
    A skipped jury stage writes no overlay, so every construct that reads that
    overlay reports n == 0 and leaves its dimension's mean — undefined, never
    a zero. Use it to stop paying a judge for a reading that is not needed.
    """
    all_stages = [
        ("extract", ["oeb.graph.extract", "--model", model]),
        ("connect", ["oeb.graph.connect", "--model", model]),
        ("assemble", ["oeb.graph.assemble"]),
        ("uptake", ["oeb.graph.juries.uptake", "--model", model]),
        ("overreach", ["oeb.graph.juries.overreach", "--model", model]),
        ("analogy", ["oeb.graph.analogy", "--model", model]),
        # the two post-graph E4 juries: reward hacking per act, and coverage
        ("reward_hacking", ["oeb.graph.juries.reward_hacking", "--model", model]),
        ("coverage", ["oeb.graph.juries.coverage", "--model", model]),
        ("measure", ["oeb.measure"]),
    ]
    skip = {n.strip() for n in os.environ.get("OEB_CHAIN_SKIP", "").split(",")
            if n.strip()}
    unknown = skip - {n for n, _ in all_stages}
    if unknown:
        raise SystemExit(f"OEB_CHAIN_SKIP names no such stage: {sorted(unknown)}")
    return [(n, a) for n, a in all_stages if n not in skip]


def core_version() -> str:
    def git(*a):
        r = subprocess.run(["git", "-C", str(CORE), *a],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    head = git("rev-parse", "--short", "HEAD")
    if head is None:
        return "unknown (not a git checkout)"
    return head + ("+dirty" if git("status", "--porcelain") else "")


def run_unit(unit: Path, model: str, retries: int, wait: int) -> bool:
    log = unit / "chain.log"

    def note(fh, msg):
        fh.write(f"=== [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        fh.flush()

    for attempt in range(1, retries + 1):
        with open(log, "a") as fh:
            note(fh, f"attempt {attempt}: core {core_version()}, judge {model}")
            for name, (mod, *rest) in stages(model):
                note(fh, name)
                rc = subprocess.run([sys.executable, "-u", "-m", mod, str(unit), *rest],
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    note(fh, f"{name} failed (exit {rc})")
                    break
            else:
                note(fh, "CHAIN DONE")
                return True
            if attempt < retries:
                note(fh, f"retry in {wait}s")
        if attempt < retries:
            time.sleep(wait)
    with open(log, "a") as fh:
        note(fh, f"CHAIN EXHAUSTED after {retries} attempts")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", nargs="+")
    ap.add_argument("--model", required=True,
                    help="the judge, e.g. gpt-5.6-sol or claude-opus-5")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--wait", type=int, default=300)
    a = ap.parse_args()
    units = [Path(u) for u in a.units]
    missing = [str(u) for u in units if not (u / "record.jsonl").is_file()]
    if missing:
        raise SystemExit(f"no record.jsonl in: {', '.join(missing)}")

    def one(u):
        ok = run_unit(u, a.model, a.retries, a.wait)
        print(f"{u}: {'done' if ok else 'EXHAUSTED'} (log {u / 'chain.log'})",
              flush=True)
        return ok
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(one, units))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
