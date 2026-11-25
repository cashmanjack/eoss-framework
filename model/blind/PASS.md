# Pass mark for the blind test (placeholder: Jack sets this)

STATUS: UNSET

Nothing below is decided. The scoring tool prints the metrics defined in `model/blind/RULE.md` and, while STATUS is UNSET, prints no verdict. Set the mark
BEFORE the first new dataset is calibrated (the commit that fills this file must come before any `calibration_*` file), otherwise the test is not pre registered.

Fill in (one line each, numbers only, then change STATUS to SET and commit):

- Primary absolute rows (RTNN, TrueKNN, Arkade; ratio = measured / predicted): a row passes when 0.5 <= ratio <= ___ (the default in RULE.md is 2).
- Primary pass rule over all rows: at least ___ of ___ rows pass, and rms ln ratio <= ___.
- Per mapping rule (optional): ___
- Relative rows (ratio of measured to predicted speedup): a row passes when ___ <= ratio <= ___.
- Relative pass rule: at least ___ of ___ rows pass.
- Secondary (uncalibrated v1, RTNN): reported only, not part of the pass rule: yes / no (change here): ___
- Datasets and K list of the test (so no row can be added or dropped afterwards): ___
