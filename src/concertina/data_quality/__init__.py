"""Offline data quality for LeRobot v3 robot datasets: scan, quality package, pause cleaning, cleaned copy.

`concertina-data-quality scan` records evidence (`scan`, `video`, `visual_motion`), `compile` turns it into a
quality package (`pipeline`, `policy`), `materialize` writes the cleaned dataset (`materialize`). Training
consumes packages through `concertina.data.quality`.
"""
