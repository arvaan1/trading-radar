# Smart Money Feed (v3)

Pulls NSE's daily bulk deal, block deal, and short-selling disclosures,
aggregates them by symbol, and layers on a known-investor watchlist with
price follow-through tracking. Built up over three passes:

- **v1:** bulk/block deals + best-effort insider trading disclosures.
- **v2:** + known-investor watchlist, investor-type tagging, short-selling.
- **v3:** + persistent local cache (history accumulates across runs
  instead of re-fetching the same window daily) + price follow-through
  for known-investor deals (did the stock actually move afterward?).

**Try it immediately:** open `index.html` - ships with seeded sample data
so you can see every section before connecting anything live.

## What's new in v3

**Persistent cache.** Previously every run asked NSE fresh for the last
90 days and threw the answer away the next day. Now `data/deals_cache.csv`
and `data/short_sell_cache.csv` accumulate real history across runs - the
first run backfills the max 365 days NSE allows in one request, and every
run after that only fetches the gap since the last cached day (a few days
at most). Over time this becomes a multi-year archive (capped at ~2 years
retention) without ever re-asking NSE for data it already has.

One deliberate design choice: only raw facts are cached (date, symbol,
client, qty, price). Fields like `investor_type` and `is_known_investor`
are recomputed fresh from your *current* `known_investors.txt` every run,
against the *full* cached history. Practical effect: add a new name to
that file, and their entire trading history in the cache gets tagged
retroactively - not just deals going forward.

**Price follow-through.** For known-investor BUY deals, this checks what
the stock actually did in the 5/10/20 trading days after, via Yahoo
Finance (same source DMA Radar uses). Aggregated per investor into a
track record: average 20-day return, hit rate. Capped at ~40 price
lookups per run to keep runtime bounded regardless of how large the cache
gets - this grows in coverage over time, not all at once.

## The combined cross-tool view

A fourth page, `../combined/index.html`, reads this tool's output plus
DMA Radar's and Delivery Radar's directly - no new scan, no NSE calls of
its own, just cross-referencing three already-published JSON files by
symbol in the browser. Shows, per symbol, whether each tool reads bullish,
bearish, or neutral, and an agreement score. Deliberately shows each
tool's raw read side by side rather than collapsing everything into one
opaque number - transparency over a single "smart" score.

## Running it yourself

```bash
pip install -r requirements.txt

python scripts/scan.py --demo    # synthetic data, no network
python scripts/scan.py           # live, incremental cache update
```

No `--lookback-days` flag anymore - the cache window is managed
automatically based on what's already stored.

## The honest data-source risk, still true in v3

Bulk/block/short-sell data still goes through NSE's interactive API (the
`nse` PyPI package, session cookies cached to disk, `server=True` mode
built for unattended use) - the same category of endpoint that failed
silently on Delivery Radar's first attempt, mitigated but not eliminated.
If a live run comes back with an unexpectedly small cache, check for
warnings in the run log the same way you would have for Delivery Radar.

## Reading the Known Investor Track Record

- **Trades w/ Data:** only counts trades where 20 trading days have
  actually elapsed since the deal - very recent trades show "pending"
  instead of a return, since there's nothing to measure yet.
- **Avg 20d Return / Hit Rate:** self-explanatory, but worth saying
  plainly - a good historical hit rate is not a promise, and the sample
  size here starts small and grows as the cache accumulates more history
  across future runs.

## Where this fits with the other tools

Same shape as always: fetch -> compute -> JSON -> static dashboard, and
now a fourth page that ties all three together without needing its own
data pipeline at all.
