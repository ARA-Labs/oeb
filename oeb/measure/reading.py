"""A reading is the ONLY numeric object in the measurement layer.

Reading(k, n): n = how many opportunities the record offered for the
behaviour, k = how many of them the agent took (for a share-valued
construct, the sum of the items' shares).

n == 0  -> the reading is UNDEFINED (the world offered no opportunity);
           it carries no rate and casts no vote.
n small -> the rate is defined but its interval is wide; the aggregate
           keeps a count-valued reading with fewer than
           aggregate.MIN_N opportunities out of the dimension mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt


@dataclass(frozen=True)
class Reading:
    construct: str
    k: float          # sum of item values; an int when every item is 0/1
    n: int
    # gids (or other item ids) behind k and n, kept so every number is
    # auditable back to the record. Not part of equality.
    taken: tuple = field(default=(), compare=False)
    opportunities: tuple = field(default=(), compare=False)
    # per-item values in [0, 1]; () means every item is 0/1 (binary).
    # A construct whose items carry a SHARE (T1: do-side token share
    # of a belief's birth budget) reports the mean share, never a
    # 0/1 vote.
    values: tuple = field(default=(), compare=False)
    # optional per-construct disclosure (a distribution over item kinds,
    # a note) — reported beside k/n, never read by the aggregate.
    detail: dict = field(default_factory=dict, compare=False)
    # a construct that declares its items as shares (a contract formula)
    # is share-valued even when a value lands on 0 or 1 exactly
    share: bool = field(default=False, compare=False)

    def __post_init__(self):
        if self.n < 0 or self.k < 0 or self.k > self.n:
            raise ValueError(f"{self.construct}: k={self.k} n={self.n}")

    @property
    def defined(self) -> bool:
        return self.n > 0

    @property
    def rate(self) -> float | None:
        return self.k / self.n if self.n else None

    @property
    def binary(self) -> bool:
        if self.share:
            return False
        return not self.values or all(v in (0, 1, 0.0, 1.0) for v in self.values)

    def interval(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson score interval for binary items; percentile bootstrap
        over items for share-valued ones; None when undefined."""
        if not self.n:
            return None
        if not self.binary:
            import random
            rng = random.Random(0); vals = list(self.values); n = len(vals)
            sims = sorted(sum(rng.choice(vals) for _ in range(n)) / n
                          for _ in range(2000))
            return (sims[50], sims[1949])
        p, n = self.k / self.n, self.n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return (max(0.0, c - h), min(1.0, c + h))

    def to_json(self) -> dict:
        lo_hi = self.interval()
        return {
            "construct": self.construct,
            "k": self.k, "n": self.n, "binary": self.binary,
            "rate": self.rate,
            "ci95": list(lo_hi) if lo_hi else None,
            "taken": list(self.taken),
            "chances": list(self.opportunities),
            **({"detail": self.detail} if self.detail else {}),
        }
