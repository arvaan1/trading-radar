# Delivery Radar

Flags NSE stocks where today's delivery percentage is a statistical outlier
against *that stock's own* trailing 20/60-day average - not a market-wide
number, because normal delivery % varies enormously by stock. A sleepy
small-cap running 60% delivery on an ordinary day and a high-beta trading
favourite running 15% are both "normal" for themselves; what this tool
looks for is either one trading well outside its own usual range.

Same free-forever pattern as the DMA Radar: a scheduled GitHub Action does
the daily compute, a static page reads the JSON it produces. This one is
even lighter - it needs only one data source (`nselib`), not two.

**Try it immediately:** open `index.html` - it ships with seeded sample
data covering all four signal types so you can see what each looks like
before connecting anything live.

## The four signals

| Signal | Delivery % | Price move | What it suggests |
|---|---|---|---|
| **Quiet accumulation** | Outlier (spike) | ~flat | Shares changing hands for real, without moving the stock or triggering a volume-spike scanner - the "someone is quietly building a position" case |
| **Confirmed accumulation** | Outlier (spike) | Up | Same real-delivery signature, but with price conviction behind it |
| **Possible distribution** | Outlier (spike) | Down | Could be genuine sellers exiting, could be value buyers absorbing supply on a dip - the tool flags it, you judge it |
| **Speculative churn** | Below-average | Volume spike | The explicit contrast case: high volume that is NOT settling into delivery, i.e. more likely to be intraday/algo round-trips than a real position change |

**Be honest with yourself about one limitation:** delivery % confirms a
trade *settled* - it does not by itself tell you who was buying and who
was selling, since every delivered share had both a buyer and a seller.
Price context is the tool's best hint at direction, not proof. Treat this
as a shortlist generator for where to look closer (results, filings,
bulk deals), not a standalone signal.

## Running it yourself

```bash
pip install -r requirements.txt

python scripts/scan.py --demo                 # synthetic data, no network
python scripts/scan.py                        # live scan
python scripts/scan.py --z-threshold 2.0       # stricter outlier bar
python scripts/scan.py --min-delivered-value-cr 1.0   # ignore thinner names
```

Serve locally to test the dashboard against real output:
```bash
python -m http.server 8000
# visit http://localhost:8000
```

## Deploying it for free

This folder lives inside the combined `trading-radars` repo alongside
`dma-radar/` — see the top-level `SETUP.md` for the full walkthrough.
This subfolder is self-contained; the workflow that keeps it updated
lives at the repo root (`.github/workflows/delivery-radar-update.yml`).
Once deployed, this dashboard is reachable at `<your-pages-url>/delivery-radar/`.

## Tuning notes

- `--z-threshold` (default 1.5): how many standard deviations above a
  stock's own baseline counts as an outlier. Raise it if you're getting
  too many low-conviction hits; the demo output at the default threshold
  typically flags a meaningful minority of a Nifty-50-sized watchlist on
  any given day - if you're seeing far more than that on real data,
  tighten this first.
- `--min-delivered-value-cr` (default 0.5): below this rupee value of
  shares actually delivered, the score gets dampened (not hidden - just
  deprioritized) so a thinly-traded stock's statistically huge but
  economically tiny spike doesn't sit at the top of your list. Toggle
  "Hide low-liquidity flags" in the dashboard to filter these out
  entirely instead.
- The 60-day z-score is a confirmation layer, not a requirement - a
  stock needs at least ~65 sessions of history for it to compute at all,
  and the score rewards agreement between the 20d and 60d windows because
  two independent baselines pointing the same way is stronger evidence
  than one.

## Where this fits with the DMA Radar

Same pipeline shape (fetch → compute → JSON → static dashboard), reused.
The natural next combination: a stock showing up on *both* radars -
approaching a golden cross AND showing quiet accumulation - is a
meaningfully stronger shortlist than either alone. Worth building as a
"combined view" once both are running for a few weeks and you trust the
individual signals.
