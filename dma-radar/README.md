# DMA Radar

A 50/200 DMA convergence scanner for NSE stocks, layered with RSI, MACD, ADX,
Bollinger bandwidth, Supertrend, volume-vs-average, and NSE delivery %.
Runs entirely on free infrastructure: GitHub Actions does the daily compute,
GitHub Pages (or any static host) serves the dashboard. No server, no
database, no monthly bill.

**Try it immediately, no setup:** open `index.html` in a browser (or serve
the folder locally, see below) — it already ships with sample data from
`python scripts/scan.py --demo` so you can see the dashboard working before
wiring up anything live.

## How it works

```
scripts/scan.py  --->  data/scan_results.json  --->  index.html (reads it)
   (runs on a                                            (static page,
    schedule via                                          zero backend)
    GitHub Actions)
```

1. `scripts/scan.py` reads `watchlist.txt`, pulls ~15 months of daily OHLCV
   per symbol from Yahoo Finance (`SYMBOL.NS`), and delivery-% history from
   NSE via the `nselib` package.
2. It computes every indicator locally with pandas/numpy (no paid data
   vendor, no TA library dependency) and scores each symbol by how
   actionable its current DMA-convergence signal is.
3. It writes the ranked output to `data/scan_results.json`.
4. `index.html` is a static page that fetches that JSON and renders the
   sortable, filterable table. That's the whole app — open the file and
   it works.

## Running it yourself

```bash
pip install -r requirements.txt

# sanity-check the pipeline with synthetic data (no network calls):
python scripts/scan.py --demo

# a real scan against your watchlist:
python scripts/scan.py

# skip the delivery-% lookup for a faster run while testing:
python scripts/scan.py --skip-delivery
```

Then serve the folder locally so `fetch()` can load the JSON (opening
`index.html` directly as a `file://` URL will hit a browser CORS block in
some browsers):

```bash
python -m http.server 8000
# visit http://localhost:8000
```

## Deploying it for free

This folder is meant to live inside the combined `trading-radars` repo
(alongside `delivery-radar/`) — see the top-level `SETUP.md` for the full
walkthrough. Short version: this subfolder is self-contained (its own
`data/`, `scripts/`, `watchlist.txt`), and the workflow that keeps it
updated lives at the repo root (`.github/workflows/dma-radar-update.yml`)
so it can commit back to `dma-radar/data/scan_results.json` on schedule.
Once deployed, this dashboard is reachable at `<your-pages-url>/dma-radar/`.

## Customizing your watchlist

Edit `watchlist.txt` — one NSE symbol per line, no `.NS` suffix. It ships
with the Nifty 50 plus a couple of seed names, but the whole point is to
point this at whatever you're actually tracking. Verify each ticker on
nseindia.com before adding it; a typo just means that symbol gets silently
skipped and logged under `errors` in the JSON output, not a crash.

## Reading the signals

- **Confirmed golden/death cross** — the 50 DMA crossed the 200 DMA within
  the last 5 sessions.
- **Approaching golden/death cross** — the gap between the two is under 2%
  and narrowing, i.e. a cross may be close. This is the "early detector"
  part — most public scanners only tell you *after* the cross happens.
- **Score** — a composite that rewards proximity to a cross, narrowing
  momentum, ADX trend strength, volume confirmation, and (for bullish
  setups) rising delivery % relative to its own recent average. It's a
  ranking aid, not a signal to trade on by itself — always sanity-check
  the underlying numbers.

## Known limitations, honestly

- NSE's data endpoints (used via `nselib`) are public but **undocumented**
  — they can rate-limit or change shape without notice. The delivery-%
  fetch is wrapped in a try/except for exactly this reason: it degrades
  gracefully (shows `—`) rather than breaking the whole scan.
- This is an end-of-day tool by design (daily DMA crossovers don't need
  intraday refresh). If you want an intraday version of the options/OI
  dashboard idea we discussed, that needs a different refresh cadence —
  worth a separate build.
- Nothing here is investment advice. It's a screening aid to narrow your
  own research, the same way you already filter pharma/hydrogen names by
  hand — this just automates the "which 10 out of 500 deserve a closer
  look today" step.

## Where this goes next

The pipeline pattern here (fetch → compute → JSON → static dashboard) is
reusable for the other product ideas from our conversation — a delivery-%
anomaly radar, a bulk/block deal + insider trading feed, an FII/DII macro
cockpit. Same skeleton, different `scan.py` logic. Ask your Claude mentor
when you're ready to build the next one.
