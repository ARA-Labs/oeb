"""oeb.measure: the measurement layer, from graph and jury overlays to dimension scores.

A construct's reading is  taken / opportunities,  both counted from the
unit's unified graph and jury overlays. Three constructs report a share
instead: T1's two (the mean do-side token share) and E4's hacks_counted
(1 / (1 + hacks)). See reading.py (the only numeric type),
constructs.py (the construct table), aggregate.py (the averaging rule
and its five-opportunity floor), events.py (loading a unit, and the
scoring pool).
"""
