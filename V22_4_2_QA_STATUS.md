# Alpha Radar V22.4.2 validation-refill QA

Changes are intentionally narrow and development-only.

1. Discipline-after-ranking refill
   - Old behavior: rank top_k, then reject RSI/bias/double-top/sector violations and leave rejected slots as cash.
   - New behavior: preserve the same filters, scan at most 4 x top_k ranked names, and refill until top_k valid names are found or the bounded scan is exhausted.
   - Report adds selected_count, refill_used, rank_depth_used and missing_slots.

2. Two-layer model ranking
   - Stock-selection score = mean excess versus the same KOSTO250 pool.
   - Policy score = min(mean excess versus 0050, extra-cost-stress excess).
   - Development ranking = 70% stock-selection + 30% policy - existing model-complexity penalty.
   - Failure blockers and final confirmation verdict remain strict and unchanged.
   - One-time pool calibration retains the original strict maximin score, so this patch does not reopen pool choice.

3. Resume/version behavior
   - Existing data, pool lock and pool study remain reusable.
   - A completed V22.4.1 fast report is treated as stale after the engine version change, so the UI allows a one-click V22.4.2 rerun without deleting data.
   - Final holdout is still not opened by guided validation.

Local checks
- Python compileall: PASS
- Module import smoke (non-Streamlit modules): PASS
- Synthetic refill test: PASS (10 funded names recovered after raw top-10 discipline rejection)
- Selection-score formula test: PASS
- JSON configuration parse: PASS

No claim of improved market performance is made until the same guided validation is rerun on the pinned dataset.
