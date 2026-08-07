# Macro Cockpit

One dashboard, checked before you look at a single stock: India VIX (with
trend), market breadth (Nifty 50 and Nifty 500 advance-decline), FII/DII
cash market flow (with trend), and participant-wise F&O index futures
positioning (FII/DII/Pro/Client - who's leaning which way).

**Try it immediately:** open `index.html` - ships with seeded sample data.

## Data source confidence, honestly

Three of the four pieces are solidly verified:

- **VIX** - official NSE historical endpoint, supports a real date range.
  Backfills 40 days immediately on first run.
- **Participant OI** - a plain daily CSV NSE publishes openly, same
  reliable pattern as Delivery Radar's bhavcopy file.
- **Advance-Decline** - official NSE endpoint, but only returns *today's*
  snapshot (no historical range) - so its 5-day/20-day trend builds up
  from a local cache over the coming days/weeks rather than backfilling
  immediately. Nothing broken about this, just a real limitation of what
  NSE exposes.

**FII/DII cash flow is the one piece built on less certainty.** The
endpoint is real and well-documented in the open-source NSE-scraping
community, but its exact response shape couldn't be verified from a
sandboxed build environment. The parsing code tries several plausible
field names and logs clearly rather than crashing if none match - if
you check the dashboard and FII/DII cash shows "—" for more than a
couple of days running, that's the thing to flag for a fix, and the run
log will show exactly what NSE actually returned.

## Running it yourself

```bash
pip install -r requirements.txt
python scripts/scan.py --demo    # synthetic data, no network
python scripts/scan.py           # live, incremental cache update
```

## Why some numbers say "building history..."

Unlike VIX (which NSE lets you backfill directly), advance-decline and
FII/DII cash only ever give you *today's* number - there's no historical
range to ask for. So the very first time you run this, only "today"
exists, and the 5-day/20-day trend fields will be empty. Every day the
scan runs after that adds one more real day to the local cache
(`data/macro_history.csv`), and the trend fields fill in on their own -
same pattern as Smart Money Feed's price follow-through feature. No
action needed, it just needs a week or so of real runs to become fully
useful.

## Where this fits with the other tools

This one is a market-level gauge, not a per-stock signal, so it's not
part of the Combined View's per-symbol table - but it's the thing worth
checking *before* that table, to decide how aggressively to act on
whatever it shows you.
