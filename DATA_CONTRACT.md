# Normalized market-data input contract

Import is optional. Default automatic mode downloads real daily amount data and adjusted prices separately. Use import for licensed historical universes, a prepared full-market dataset, market-cap snapshots or optional point-in-time factors. Never use a previous results JSON as market data.

The Streamlit import ZIP must have only the following allowlisted files at its ROOT, no directories, nested ZIP, Python source, absolute paths or symlinks. Uncompressed ZIP members must total no more than 1 GB. The default Streamlit upload limit still applies. For larger files, use the CLI with a local directory. `prices.csv.gz` is supported to reduce transfer size.

## 1. manifest.json (required)

```json
{
  "data_kind": "PROVIDER_HISTORY",
  "amount_kind": "ACTUAL_TRADED_VALUE",
  "amount_unit": "TWD",
  "volume_unit": "SHARES",
  "price_basis": "ADJUSTED_PROVIDER",
  "source": "Your licensed provider, dataset name, extraction timestamp and vintage details",
  "membership_scope": "HISTORICAL_MEMBERSHIP"
}
```

For current-snapshot import, declare `membership_scope="CURRENT_SNAPSHOT"` and leave requested_scope at CURRENT_OFFICIAL_SNAPSHOT. Only the explicit synthetic fixture is exempt from the minimum roster sanity check. Do not set SYNTHETIC on real data to bypass checks. All imported provenance is USER SUPPLIED, not independently verified by this app.

## 2. catalog.json (required)

An array of all source securities, excluding the benchmark:

```json
[
  {"ticker":"2330.TW","name":"Example name","board":"TWSE","sector":"Example sector"},
  {"ticker":"6488.TWO","name":"Example name","board":"TPEX","sector":"Example sector"}
]
```

These two rows show SCHEMA ONLY and are not a usable market universe. A real dataset needs the entire declared catalog (default sanity check >=1500 distinct securities). Stock identifiers use the ordinary four-digit Taiwan code plus `.TW` or `.TWO`. The catalog cannot contain 0050, warrants, ETFs, repeated symbols or fabricated missing-company substitutes. For historical mode it must include the union of all relevant historical securities; do not silently omit delisted companies. Listing-board transfers need a consistent vendor symbol identity, monthly memberships and raw histories. The importer does not reconstruct those events for you.

## 3. calendar.csv (required)

```csv
date
2018-01-02
2018-01-03
```

Use actual common-market sessions, not an invented Monday-Friday calendar. Each date occurs once. It must cover raw input history plus warmup and complete evaluation exits. Monthly signal dates are calculated from this calendar; no stock's next available observation can silently redefine a missing exchange session.

## 4. prices.csv OR prices.csv.gz (required)

```csv
ticker,date,raw_open,raw_high,raw_low,raw_close,volume,amount,adj_open,adj_high,adj_low,adj_close
```

- Required for every source-catalog security and for `0050.TW`.
- One row per (ticker,date), no duplicate date or revision rows.
- Raw OHLC: actual historical unadjusted prices in TWD.
- `volume`: actual shares, NOT lots, thousands of shares, or reconstructed adjusted volume.
- `amount`: actual reported traded value in TWD, NOT adjusted close times volume, raw close times volume, or an estimate. For FinMind this is `Trading_money`.
- Adjusted OHLC: consistently adjusted provider prices used for return comparison and ratios. Do not mix raw endpoints, different adjustment vintages or differently scaled OHLC within one price series.
- Confirmed actual zero activity stays zero. Unavailable values stay blank/NaN. Do not turn an absent source row into a zero-trading day, forward fill a suspended exit or invent a benchmark return.
- Data must support trailing252 amount ranks for at least the largest requested pool at each predetermined month. Real new listings need not have a full year, but they cannot qualify as one-year histories before it exists.
- No future labels, predicted returns or prior strategy scores are required. They are computed by the engine after the date-T membership and ranks are fixed.

The first import is chunked into 100,000-row transactions with atomic raw/adjusted/resume markers. Later phases use a symbol at a time; do not confuse an imported dataset's full file size with the factor engine's peak memory.

## 5. membership.csv (required for historical-membership mode)

```csv
date,ticker,available_at,source
2020-01-31,2330.TW,2020-01-31T13:30:00+08:00,provider_snapshot_reference
```

Supply every ordinary stock in the declared historical universe at EACH monthly signal date, not just its Top-500 survivors. Snapshots use exact month-end dates. `available_at` must not be later than that date's 23:59 Asia/Taipei. The code checks schema, chronology and duplicates; it cannot certify the completeness of your snapshots. Metadata for names/industries is not used to pretend current industry labels are historical truth.

## 6. market_caps.csv (optional)

Enable `include_market_cap_pools=true` only with complete dated snapshots:

```csv
date,ticker,market_cap,available_at,source
2020-01-31,2330.TW,10000000000000,2020-01-31T13:30:00+08:00,provider_reference
```

The number above is SCHEMA ONLY, not an actual historical valuation. `market_cap` is TWD total market capitalization at the date, available by date-T. Market-cap comparisons use the same raw-turnover-eligible securities and dates as amount comparisons. A missing cap for an eligible security blocks cap ranking; do not assume it is small. Current shares times old prices, paid-in capital, and current market cap backfilled through history are NOT acceptable substitutes. Optional mode adds MCAP100/200/300/500 alongside ADV100/200/300/500.

## 7. evidence.csv (optional point-in-time factor input)

Enable `enable_optional_factors=true` and provide:

```csv
ticker,available_at,factor,value,source
2330.TW,2020-02-10T18:00:00+08:00,revenue_yoy,0.15,release_reference
```

This is a schema example, not an actual announcement or factor observation.

Allowed factor names: `revenue_yoy`, `eps_growth`, `foreign_flow20`, `trust_flow20`. Supply precomputed comparable factor values from information genuinely available at that timestamp, with consistent units across securities. The code does NOT compute or verify these four economic definitions from financial statements/flow raw data. It ranks the supplied values; raw signed institutional share counts would create a size exposure and should not be mislabeled normalized accumulation.

The as-of join is backward only, at signal-date 23:59 Asia/Taipei, with a fixed 120-calendar-day freshness limit. Revised data needs its own later availability timestamp; never replace earlier vintage values as if known earlier. All four factor columns exist, missing values remain NaN and candidates must pass the fixed coverage rules. Factors not supplied or insufficiently observed are not imputed to zero and do not become imaginary evidence.

## Common blocks

- A 150-stock results report is not a full-market annual amount history.
- A 500-name current list alone cannot define the historical Top 500.
- A current roster is not a historical membership ledger.
- Complete file download does not imply complete selected future outcomes.
- Market-cap/financial comparisons remain disabled unless their dated input is provided.
- Too-short history or too-few valid folds blocks the experiment; the engine will not lower its minimums or reuse dates to obtain a positive verdict.
