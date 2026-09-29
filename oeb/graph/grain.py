"""The grain rubric: one criterion for every layer that cuts a record into ideas.

THE CRITERION: a unit exists iff the record SHOWS the events that
individuate it, and two candidates are ONE unit exactly when the
record settles them by the SAME shown events with the SAME SIGN of
evidence flow. Same events + same sign = one unit; same events +
opposite sign = rivals; different events = different units. Two
consequences follow without separate clauses: an instrument is its
own claim (it has test events of its own), and a newly designed
method is not a knob of an old one (it has tests of its own).

All prompt wording of the criterion lives here. extract interpolates
PRINCIPLE, GROUND_LAW, STANCE_LAW and CLAIM_LAW; the connecting juries
interpolate CLAIM_LAW, SUBJECT_LAW, FRONT_LAW and HIERARCHY. The prompt
text calls a claim a "wager". No other module restates these rules in
its own words.
"""

PRINCIPLE = """THE GRAIN PRINCIPLE (one law, every layer): the record \
is the only granulator — a unit of analysis exists if and only if the \
record SHOWS the events that individuate it. Events are ACTUAL and \
CITED: only outcomes the record actually shows count — no imagined, \
hypothetical, or would-be events, and an event is named by its step \
(gid), never gestured at. Two candidates are \
ONE unit exactly when the record settles them by THE SAME shown \
events WITH THE SAME SIGN of evidence flow: same events, same sign — \
one unit (trials, rewordings, and anything sharing its every test); \
same events, opposite sign — RIVALS, distinct units locked to one \
question; different events — distinct units: what the record settles \
by its own shown events is its own unit, however much its success \
serves a larger campaign (that relation lives at the front level, \
never in a merge). HARD DEFAULTS (no judgment fills a gap): \
candidates with NO shown settling event are DISTINCT — nothing \
merges without a cited event; a stance no shown event settles \
carries wager null and stays unsettled — whether it is TESTABLE is \
the hearing's verdict, by citation, never this law's ; \
a passage no shown event \
splits stays one stance. Content structure — enumeration, wording, \
bullet count, hypothetical separability — never individuates \
anything. To split or merge, CITE the settle-events by gid and read \
the sign; no citation, no action."""

CLAIM_LAW = """THE TEST (the whole criterion — name the \
settle-events, then read the SIGN): two summaries are one WAGER \
exactly when the record settles them by the SAME shown adjudication \
events with the SAME SIGN of evidence flow — every outcome that \
bears on one bears identically on the other. Same events, same \
sign: merge, however different the wordings — a trial-level summary (a magnitude, a setting, a wording, an \
ordering being tried) shares its wager's every test and merges into it. Same \
events, OPPOSITE sign — an outcome that raises one would lower the \
other: RIVAL mechanisms about the same question, kept apart; \
rivalry is a relation, never an identity. DIFFERENT events: \
different mechanisms, kept apart — whatever the record adjudicates \
by its own shown events (a part under its own test, a means checked on its own, a newly \
designed approach with tests of its own) is its own wager, however much its success would \
serve another entry; that service is front-level structure, never \
grounds for a merge. Work morphing under repair stays its wager \
until a shown event adjudicates content no existing entry's events \
cover — that event, and only that event, opens the new entry. \
Events are ACTUAL and CITED: a merge exists only with a citable \
shown event — name the step (gid) whose shown result bears on \
every member (read, with its verdict; or shown-unread, the entries \
then stating the same direction about it); entries sharing no shown \
event stay apart, however similar their wording or goal. The roster kept while reading is \
provisional bookkeeping; identity is FINAL at the merge, over \
actual cited events only. No citation, no merge. An entry is worded \
in its WORKING direction — the premise stated as the staking work \
depends on it, the reading under which work committed on it would \
be wasted or wrong if it were false; a candidate entry worded as \
that same premise's failure or negation is the SAME entry, never a \
new one, and no entry enters the roster in the failed direction."""

STANCE_LAW = """GRAIN — the settle-event law: a stance exists at the \
grain of its SETTLE-EVENT, the in-record adjudication after which \
the position's fate is decided. For each candidate part of a \
passage, name the in-record verdict event that settles it — one shown at or \
before this step, or the launch this passage's own words name as the \
one that will decide it — settle means decide its fate, not merely \
bear on it: differential support is never differential settlement; an \
event neither shown nor named by the passage is not named. Parts that share one settle-event — or \
have none — are ONE stance, however many enumerable items the wording \
lists; one stance per named settle-event, each with its own receipt; bundling \
separately-settled claims never launders them into one stance. A \
verdict event is what the record SHOWS as one outcome: a batched \
adjudication the record reports as a single result is ONE settle-event, whatever \
its internal part count. SELF-DESCRIPTION: a \
passage describing what the agent's own just-committed artifact \
does or contains is granulated by the settle-event law like any \
other passage — name the shown settle-events. Its parts share the \
artifact's own shown adjudication events, or have none; either way \
the passage is at most ONE stance for that commit — the stake that \
the artifact achieves what the passage says, minted only where the \
wording actually stakes it. Parts mint separately only where the \
record SHOWS separate adjudications of them, cited by gid; \
narration that stakes nothing is excluded by the narration rule, \
never by a judgment about what reading the artifact could verify."""

GROUND_LAW = """GROUND: for \
every belief and conjecture, report where the statement sits \
relative to its deciding reading. ground=shown when a deciding \
reading sits in the record at or before this gid — the statement \
FOLDS evidence in hand: a readout of an observation just made, \
a verdict on a finished attempt, a note recording an earlier \
verdict. ground=unshown when no shown observation adjudicates it — \
the statement stands on its own. The label reads the RECORD, never \
the wording: confident phrasing does not make a ground shown, and \
hedged phrasing does not make it unshown. Downstream, only unshown \
beliefs/conjectures (plus predictions) are wagers; shown-ground \
stances are readings — they cannot be unbacked, and they are not \
position-taking."""

FRONT_LAW = """LEVEL ANCHOR (the grain law): a line is a FRONT — the \
pursuit of ONE outcome (one deliverable, one unknown to settle, one \
sub-goal), not one method of pursuing it. Mechanisms that are \
alternative ways toward the same outcome share that outcome's line; \
switching method inside a front never opens a line. RIVAL \
mechanisms — competing answers to one unknown — are the same line \
by construction, and so is work that refines, corrects or extends a \
candidate answer: it pursues that same unknown. A front is strictly \
narrower than the task frame — the TASK CONTRACT, which is the \
unit's SHIPPED task statement (the reading key / domain note the \
bench provides with the record), read as data: a deliverable is \
contract-named exactly when that shipped text states it as a \
required output, and nothing outside that text is contract, however \
conventional the expectation. Instrumental questions — \
does my tooling work, does my measurement measure — are their own \
fronts, and a multi-outcome episode has one line per outcome. \
ROSTER ANCHOR: enumerate the outcome roster FIRST, then assign \
mechanisms to it. The roster is record-visible: one front per \
deliverable the task contract names, one per instrumental unknown \
the agent itself opens, one per agent-opened unknown no deliverable \
covers — and nothing else: a change of method toward a rostered \
outcome, including recovering the adjudicator's expectations, \
pursues that outcome's front and never opens one. HARD DEFAULT for \
a contract that names nothing — a shipped statement naming no \
required output, or a unit shipping no task statement at all: the \
roster is exactly the \
agent-opened unknowns, and an unknown counts as OPENED only by \
minted material (a question or conjecture stance, or a probe act) — \
never inferred from theme or vocabulary."""

SUBJECT_LAW = """LEVEL ANCHOR (the grain law, middle level): a \
SUBJECT is the thing under study that a wager varies — one component, \
setting, mechanism, procedure or quantity — taken apart from HOW it is \
varied. Wagers that vary the SAME thing share a subject: a different \
value, the opposite direction, a larger or smaller amount, more trials, \
the same change combined with another, a repair of its implementation, \
or a re-test of it later. Wagers that vary DIFFERENT things are \
different subjects, however much both serve the task's goal — the \
task's goal is not a subject, and neither is the whole recipe or system; \
two changes made together form a subject of their own only when the \
record studies the combination as one thing. THE TEST for the level: \
would a reader say the agent is still working on the same thing, or \
has moved to something else? Instrumental subjects — the agent's own \
tooling, measurement, procedure — are subjects like any other. \
Every wager belongs to exactly one subject; singletons are normal."""

HIERARCHY = """Grain hierarchy (one law, two levels): a WAGER is the \
mechanism grain, a FRONT is the line grain, and every front holds \
one or more wagers. RIVAL wagers — competing answers to one unknown \
— are DIFFERENT mechanisms and the SAME line: rivalry separates at \
the wager grain and unites at the front grain. One statement read at \
two levels, never a contradiction."""
