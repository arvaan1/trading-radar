# Re-Rating Radar

A pre-re-rating detector - built to catch a stock somewhere between
"quietly improving" and "the market has fully noticed," not after.

**Read this before trusting its ranking the way you'd trust DMA Radar's
or Delivery Radar's:** this tool is built from one deeply-researched,
real case study, not a statistically backtested model. That distinction
is repeated on its own dashboard, on purpose - it's the difference
between honest and dishonest about what this actually is.

## Where this came from

The original ask behind this tool was a genuinely ambitious research
brief: reverse-engineer the "anatomy" of major Indian-stock re-ratings
(Cupid, Kernex Microsystems, Paras Defence, CEMPRO, KMEW) using
forensic, point-in-time analysis, then build three separately-validated
screener methodologies with statistically-derived weights, backtested
across the wider NSE universe across multiple market regimes.

That full brief needs data this system genuinely doesn't have access to
- quarterly fundamental time series, order-book history, and
shareholding *changes* over time, for hundreds of companies across
years. No free NSE endpoint publishes that in bulk, structured form.
Building the full brief anyway, using whatever could be scraped
together, would have meant dressing up a guess in the language of
statistical rigor - "backtested," "hit rate," "statistically derived
weight" - without any of those words actually being earned. That's
worse than being upfront about the limit.

## What's real here instead

One real, cited case study: **Paras Defence and Space Technologies**
(NSE: PARAS). Documented facts, not inference:

- Net sales grew from ~Rs 143 Cr (FY21) to ~Rs 365 Cr (FY25), with debt
  reduced to zero - a multi-year fundamental trajectory that predates
  the stock's explosive move by years, not weeks.
- Independent technical analysis documented a ~110-day low-volume price
  consolidation ("coiling") through mid-2025 to early 2026, followed by
  a breakout on volume roughly 10x its own 20-day average.
- The subsequent acceleration was driven by specific, repeated order-win
  announcements (a Rs 142 Cr MoD/DRDO order in March 2025, a Rs 53 Cr
  BEL order in June 2026) - these are lagging/confirming signals by
  definition, since you can't front-run a filing that hasn't happened
  yet, but a company already capable of winning repeat high-value orders
  is itself a detectable trait beforehand.

## What this tool actually screens for

Every input below is data this system already computes reliably -
nothing new or fragile:

- **Compression** - is this stock's own Bollinger bandwidth (already
  computed by DMA Radar) currently in the tightest quarter of its own
  recent range? Tracked via a small persistent cache this tool builds
  itself, run over run - starts empty, needs ~10 days of real runs
  before it says anything meaningful, same honest "grows over time"
  pattern as every other cache in this system.
- **Quiet outperformance** - positive Relative Strength (DMA Radar), but
  not yet extreme. The scoring sweet spot is +2% to +20% - already
  leading, not yet the story everyone's telling. RS above 40% scores
  nothing here on purpose; that's likely already a later-stage move.
- **Real accumulation** - Delivery Radar's own 20-day delivery z-score,
  reused directly.
- **Informed participation** - a tracked known investor or a repeat
  buyer, from Smart Money Feed.
- **The transition itself** - was the stock compressed, and has volume
  just started expanding (>1.3x its 20-day average)? That's the
  specific state-shift this tool exists to catch.

## The score is transparent, not fitted

Every point in the 0-100 score is explained in `scripts/scan.py`'s
`score_row()` function with the reasoning behind it - there's no hidden
weighting or claim of statistical optimization. Verified against three
constructed test cases before shipping: a textbook Paras-shaped setup
scored 77/100; the same setup after it's already broken out and become
obvious scored 5/100 (correctly deprioritized, on purpose - that's the
whole point of this tool); a boring, nothing-happening stock scored 0.

## Running it yourself

```bash
pip install -r requirements.txt
python scripts/scan.py --demo    # synthetic data, no network
python scripts/scan.py           # live - reads DMA/Delivery/Smart Money's
                                  # own already-published output
```

Run this AFTER DMA Radar, Delivery Radar, and Smart Money Feed each
evening - it has nothing to say without their output.

## What would make this genuinely trustworthy over time

Two things, neither achievable in a single build: (1) the compression
cache needs real weeks of accumulated history before its percentiles
mean anything, and (2) the only real test of this methodology is
watching what it actually flags, over real time, against what actually
happens next - the same self-backtesting pattern DMA Radar and Delivery
Radar already do for their own signals. That's a natural next step once
this has real history to check itself against - not something honest to
claim on day one.

## What was researched but not built

Kernex Microsystems, Cupid, and CEMPRO's more recent trajectory (as
Cemindia Projects, post-Adani acquisition) weren't researched to the
same depth as Paras Defence in this pass, given the scope already
covered. Worth a dedicated follow-up if you want the full five-stock
picture this tool's design could then be checked against.
