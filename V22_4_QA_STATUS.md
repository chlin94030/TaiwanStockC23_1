# V22.4.1 QA status

Release: `22.4.1-guided-derived-pool-lock`

## This fix
- Fixes PRECHECK failure: `Requested pool not present in pinned snapshot`.
- KOSTO250/KOSTO300 are V22 deterministic pools derived from the pinned ADV500 snapshot, so they now use a V22-specific integrity lock instead of the legacy native-pool lock.
- The lock is anchored to dataset id, V22 feature id, derived monthly-membership SHA and source monthly-pool SHA.
- Stale pool-study suggestions from a different dataset are ignored automatically.
- Existing data, pool-study results and checkpoints are preserved; no database rebuild is required.
- Legacy `null` cache compatibility from V22.4 remains in place.

## Automated QA
- Full repository test suite: **202 passed, 0 failed**.
- Added regression test proving KOSTO250 can be frozen when it is derived by V22 but not a native pool label in the pinned snapshot.
- Added regression test preventing reuse of a pool suggestion from a different dataset.
- Python compile checks passed.

## Limits
- No real Streamlit Cloud deployment was performed from this container.
- No new real-market profitability test was run for this workflow-only fix.
- Fast validation remains exposed development evidence; the final holdout is still excluded until explicit final confirmation.
