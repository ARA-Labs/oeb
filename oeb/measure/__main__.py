"""python -m oeb.measure <unit-dir>... [--constructs]  -> writes <unit>/measure.json

Prints each unit's dimension scores (E1-E4, T1-T6) on [0, 1]; --constructs also prints
every construct's taken / opportunities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .aggregate import check_schema, dimension
from .constructs import REGISTRY, SCHEMA, SCORED
from .events import load
from .reading import Reading


def measure(unit: Path) -> dict:
    u = load(unit)
    readings = {}
    for c in REGISTRY:
        items = c.items(u) if u.status_ok else []
        vals = tuple(float(t) for _, t in items)
        binary = (not c.share) and all(v in (0.0, 1.0) for v in vals)
        readings[c.id] = Reading(c.id, sum(vals) if not binary else int(sum(vals)),
                                 len(items),
                                 taken=tuple(i for i, t in items if t),
                                 opportunities=tuple(i for i, _ in items),
                                 values=() if binary else vals,
                                 share=c.share,
                                 detail=(getattr(c.items, "detail", None) or (lambda u: {}))(u) if u.status_ok else {})
    out = {"version": "measure_v0", "unit": str(unit),
           "graph_status": u.graph.get("status"),
           "dimensions": {d: dimension([readings[c] for c in cons], scored=SCORED)
                          for d, cons in SCHEMA.items()}}
    return out


def main(argv):
    check_schema(SCHEMA)
    detail = "--constructs" in argv
    units = [Path(a) for a in argv if not a.startswith("--")]
    rows = []
    for unit in units:
        m = measure(unit)
        (unit / "measure.json").write_text(json.dumps(m, indent=1))
        rows.append((unit.name, m))
    dims = sorted(SCHEMA)
    print(f"{'unit':14s}" + "".join(f" {d:>5s}" for d in dims))
    for name, m in rows:
        line = f"{name:14s}"
        for d in dims:
            s = m["dimensions"][d]["score"]
            line += f" {s:5.2f}" if s is not None else "    --"
        print(line)
    if detail:
        print()
        for d in dims:
            print(f"== {d} constructs (taken/opportunities) ==")
            for name, m in rows:
                cs = m["dimensions"][d]["constructs"]
                print(f"{name:14s} " + "  ".join(
                    (f"{c}={v['k']}/{v['n']}" if v.get('binary', True)
                     else f"{c}=mean {v['rate']:.2f} over {v['n']}") for c, v in cs.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
