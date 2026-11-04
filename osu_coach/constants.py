"""Fixed thresholds. Kept constant on purpose so runs stay comparable over time."""

COMFORT = 93.0          # accuracy that counts as comfortable
PUSH = (85.0, 92.0)     # accuracy band that counts as a push
MIN_N = 3               # plays a bin needs before it counts
MIN_HITS = 50           # objects an attempt needs before its accuracy means anything
ERA_OLD, ERA_NEW = "old", "new"
SESSION_GAP_MIN = 30    # a gap this long starts a new session
ALPHA = 0.05            # significance level for single tests
FDR = 0.10              # false discovery rate for the weakness scan
