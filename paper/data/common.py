"""Helpers shared by the paper's generators: PostTrainBench panels, the bench's cheating verdicts, rescoring pairs.

The rescoring pairs are PostTrainBench records scored twice by GLM-5.3 on the same code, each time from an empty call
log (calls.jsonl), so no answer is replayed. They live in three folders under <root>/out/posttrainbench:
  rejudge0918-w5-glm53, rejudge0920-w5-glm53  a fixed list of (task, unit) cells, each folder <task>-<unit> paired
                                              with the panel's <unit>;
  rejudge0922-w5-glm53                        the remaining cells: <task>-<unit> paired with the panel's <unit>, or,
                                              where a <task>-<unit>-cur folder exists, with that folder instead (the
                                              panel's scoring of the record, redone on the same code as the pair's
                                              other side).
A pair counts once both sides have a measure.json.
"""
import json
from pathlib import Path

PAN = {"aime": "aime2-w5-merge-glm53", "arenahard": "arenahard-gemma-w5-glm53", "bfcl": "bfcl-gemma-w5-glm53",
       "healthbench": "healthbench-gemma-w5-glm53", "humaneval": "humaneval-gemma-w5-glm53", "gpqa": "gpqa-gemma-w5-glm53"}
FIXED = {"rejudge0918-w5-glm53": [("humaneval", "fable5"), ("humaneval", "gpt55"), ("humaneval", "kimik3"),
                                  ("gpqa", "glm52"), ("gpqa", "gpt55"), ("gpqa", "kimik3")],
         "rejudge0920-w5-glm53": [("bfcl", "opus5-r2"), ("bfcl", "fable5-r2"), ("bfcl", "kimik3-r2"),
                                  ("bfcl", "glm52-r2"), ("healthbench", "gpt56-r2"), ("healthbench", "kimik3-r2")]}
CELLS = "rejudge0922-w5-glm53"


def official_cheat(unit):
    """PostTrainBench's own verdict for the run: test items in the training data or a disallowed model.
    None when the bench shipped no verdict."""
    o = Path(unit) / "official"
    js = sorted(o.glob("judgement*.json"))
    if js:
        j = json.loads(js[0].read_text()); return bool(j.get("contamination") or j.get("disallowed_model"))
    c, m = o / "contamination_judgement.txt", o / "disallowed_model_judgement.txt"
    if c.exists() and m.exists():
        return not (c.read_text().strip().lower().startswith("no contamination")
                    and m.read_text().strip().lower().startswith("only allowed"))
    return None


def same_judge_pairs(pt):
    """[(first scoring, second scoring)] as unit folders under pt = <root>/out/posttrainbench."""
    pt = Path(pt)
    out = [(pt / PAN[t] / u, pt / b / f"{t}-{u}") for b, tu in FIXED.items() for t, u in tu]
    d = pt / CELLS
    if d.is_dir():
        names = sorted(p.name for p in d.iterdir() if p.is_dir())
        cur = {n[:-len("-cur")] for n in names if n.endswith("-cur")}
        for n in names:
            if n.endswith("-cur"): continue
            t, u = n.split("-", 1)
            out.append((d / n, d / f"{n}-cur") if n in cur else (pt / PAN[t] / u, d / n))
    return [(a, b) for a, b in out if (a / "measure.json").exists() and (b / "measure.json").exists()]
