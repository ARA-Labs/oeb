"""Aggregation: dimension = mean of its constructs' rates.

One function, no per-dimension code. Rules:
  * a construct with an undefined reading (n == 0) is absent from the
    mean — it is not 0 and not 1;
  * a count-valued construct with fewer than MIN_N opportunities, and a
    reference member (scored=False), is reported but kept out of the
    mean; share-valued readings have no floor;
  * a dimension with no construct left in the mean is undefined;
  * uncertainty is propagated by parametric bootstrap on each
    construct's binomial (k, n), or by resampling the items of a
    share-valued construct;
  * a construct belongs to exactly one dimension (check_schema), so no
    reading is averaged twice.
"""
from __future__ import annotations

import random
from statistics import mean

from .reading import Reading


def check_schema(schema: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for dim, cons in schema.items():
        for c in cons:
            if c in seen:
                raise ValueError(
                    f"construct {c!r} listed under both {seen[c]} and {dim}")
            seen[c] = dim


MIN_N = 5      # denominator floor: fewer opportunities than this -> reported, not scored


def dimension(readings: list[Reading], *, boots: int = 2000,
              seed: int = 0, scored: dict | None = None) -> dict:
    scored = scored or {}
    # a member with fewer than MIN_N opportunities is UNDERPOWERED: reported
    # (rate, CI) but kept out of the dimension mean — a 1/1 must not weigh
    # as much as an 88/130.
    # a REFERENCE member (scored=False) is reported the same way and never
    # weighs.
    # the floor is a floor on COUNTS: a share-valued reading (binary=False,
    # e.g. a two-ended band whose ends must agree) is defined by its own
    # rule and is not a 2-opportunity binomial
    def _powered(r):
        return (not r.binary) or r.n >= MIN_N
    reference = [r for r in readings if not scored.get(r.construct, True)]
    defined = [r for r in readings if r.defined and _powered(r) and scored.get(r.construct, True)]
    out = {
        "n_constructs": len(readings),
        "n_defined": len(defined),
        "undefined": [r.construct for r in readings if not r.defined],
        "underpowered": [f"{r.construct} ({r.k}/{r.n})" for r in readings if r.defined and not _powered(r) and scored.get(r.construct, True)],
        "reference": [r.construct for r in reference],
        "constructs": {r.construct: r.to_json() for r in readings},
    }
    if not defined:
        out.update(score=None, ci95=None)
        return out
    out["score"] = mean(r.rate for r in defined)
    rng = random.Random(seed)
    sims = []
    for _ in range(boots):
        acc = 0.0
        for r in defined:
            if r.binary:
                # posterior-ish draw: Beta(k+1, n-k+1); cheap and
                # honest about n
                p = rng.betavariate(r.k + 1, r.n - r.k + 1)
            else:
                # share-valued items: resample the items
                vals = r.values
                p = sum(rng.choice(vals) for _ in vals) / len(vals)
            acc += p
        sims.append(acc / len(defined))
    sims.sort()
    out["ci95"] = [sims[int(0.025 * boots)], sims[int(0.975 * boots) - 1]]
    return out
