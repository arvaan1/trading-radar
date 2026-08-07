# Options Radar

A live-ish snapshot of Nifty and Bank Nifty options positioning: PCR,
max pain, and an OI heatmap by strike - support/resistance sourced from
actual money, not chart lines. Refreshed every few minutes during market
hours, not once a day like every other tool in this repo.

**Try it immediately:** open `index.html` - ships with seeded sample data.

## Why this one works differently

Every other scanner here runs once, after market close. Options
positioning changes all session long, so a once-daily snapshot would be
stale by definition. This needs real intraday refresh - and that's
exactly the kind of job GitHub's own `schedule:` trigger is bad at, both
because it can't reliably do sub-hour intervals and because (as the rest
of this repo's setup history shows) its timing isn't trustworthy even for
once-daily jobs.

So this workflow has **no `schedule:` trigger at all** - only
`workflow_dispatch`. It's meant to be triggered externally, the same way
you set up for the same-day 11 PM deadline on the other tools: an
external service (cron-job.org) calling GitHub's dispatch API directly,
every 5-10 minutes, only during market hours.

## Setting up the intraday trigger

Same pattern as before - if you already have your GitHub access key (PAT)
and cron-job.org account, you just need **one more job**:

- **Title:** `Options Radar - intraday`
- **URL:** `https://api.github.com/repos/arvaan1/trading-radar/actions/workflows/options-radar-update.yml/dispatches`
- **Schedule:** use the "Every X minutes" option (not "Every day at") -
  set it to every 10 minutes
- **Same headers and body as every other job:** `Authorization: Bearer
  YOUR_KEY`, `Accept: application/vnd.github+json`, body `{"ref":"main"}`

One thing worth knowing: cron-job.org's "every X minutes" option runs
24/7, not just market hours - outside 9:15 AM-3:30 PM IST it'll just keep
re-fetching the same closed-market snapshot repeatedly, which is harmless
(no new commit happens if nothing changed) but slightly wasteful. If
you'd rather restrict it to market hours only, use the "Custom" schedule
option instead and enter this directly (cron-job.org runs this in your
account's timezone, which should already be Asia/Kolkata from your
earlier setup):
```
*/10 9-15 * * 1-5
```

## Running it yourself

```bash
pip install -r requirements.txt
python scripts/scan.py --demo    # synthetic data, no network
python scripts/scan.py           # live snapshot
```

## How the intraday history works

Unlike every other tool's cache, `data/intraday_history.csv` resets each
day on purpose - "how did today's PCR evolve" is the point, not a
multi-day history. Each run appends one row per symbol; the dashboard's
"Through Today's Session" charts read straight from this file.

## Reading the heatmap

- **Red bars (right side reversed) = Call OI, teal bars = Put OI**, both
  by strike, near-the-money strikes only (the far wings of a chain
  usually carry negligible OI and just add clutter).
- The **highlighted row is the ATM strike** - current spot price, roughly.
- **Max Pain** is the strike where option writers (sellers) would lose
  the least if price settled there at expiry. Price often gravitates
  toward it in the final days before expiry, especially in quiet weeks -
  not a law, just a tendency worth knowing about.

## Where this fits with the other tools

This one and Macro Cockpit are both market-level, not per-stock, so they
don't slot into the Combined View's per-symbol table the way DMA/Delivery/
Smart Money do. Instead, both feed a small "Market Context" strip at the
top of the Combined View - the weather report you check before trusting
what the per-symbol table tells you.
