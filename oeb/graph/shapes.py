"""Declared shapes for the dicts the pipeline passes around.

The data flowing between stages is raw dicts; these TypedDicts write
the contracts down where type-checkers and editors can see them.
ANNOTATION ONLY — nothing here changes runtime behavior, nothing
validates at runtime, total=False throughout (fields appear per
instrument; absence is normal, not an error).
"""
from __future__ import annotations

from typing import TypedDict


class ActRow(TypedDict, total=False):
    """One epistemic act — acts.json `acts[]` row (extract, L1)
    and, with the graph extras, a unified-graph ACT node."""
    act: str                 # check | probe | commit | revise
    gid: str                 # step id, fractional for split steps
    mech: str                # the wager (claim) the act rides (free text)
    expectation: str
    outcome: str             # pass | fail | mixed — checks only
    action_receipt: str      # verified quote of the action
    obs_receipt: str         # verified quote of the observation
    mandated: bool           # adapter-declared scaffold-forced move
    harness_event: bool
    # unified-graph extras (assemble)
    type: str                # "ACT" on graph nodes
    mech_group: str          # mech-merge jury's semantic group id


class StanceRow(TypedDict, total=False):
    """One stance — acts.json `stances[]` row / graph STANCE node."""
    id: str
    gid: str
    stance: str              # belief | conjecture | prediction | question
    #                        # | reliance | retract | attribution
    text: str
    labels: dict             # incl. ground: shown | unshown
    wager_mech: str          # roster key (the claim) whose credibility it moves
    type: str                # "STANCE" on graph nodes


class DecidingCheckCite(TypedDict, total=False):
    """One settle-event citation inside a settlement entry."""
    act: str                 # act id ("a<gid>.<ty>")
    relation: str            # confirms | contradicts
    direction: str           # same | opposite: does the relation follow the check's outcome (pass -> confirms)
    channel: str             # birth_evidence when the birth-evidence jury cited it


class SettlementEntry(TypedDict, total=False):
    """One settlement-map value, as the settlement juries write it."""
    settlers: list[DecidingCheckCite]
    testable: bool
    testability: str         # settled | quest_open | unsettled_open | aside


# stance id -> its entry. Every settlement iterator (apply_settlements,
# polarity_law) walks these per-stance entries.
SettlementMap = dict[str, SettlementEntry]
