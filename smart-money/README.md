# Smart Money Feed

Pulls NSE's daily bulk deal and block deal disclosures (large trades by
institutions/HNIs, publicly disclosed the same evening) and aggregates
them by symbol, so a stock getting repeated attention stands out instead
of being buried in a long chronological list. Also includes a best-effort
insider trading section.

**Try it immediately:** open `index.html` — ships with seeded sample data
covering a repeat-buyer pattern so you can see what that looks like before
connecting anything live.

## What "bulk" and "block" deals actually are

- **Bulk deal:** any single client's trades in a stock on one day add up
  to 0.5%+ of that company's total shares.
- **Block deal:** a single trade of at least 5 lakh shares or ₹5 crore,
  executed in a special 35-minute morning trading window, disclosed the
  same day.

Both are legally required disclosures — this tool just collects and
organizes what NSE already publishes, rather than you checking manually.

## The honest data-source risk on this one

DMA Radar and Delivery Radar both ended up on plain file downloads NSE
publishes openly — the most reliable kind of source. This tool can't do
that: bulk/block deal history is only available through NSE's interactive
API, the same category of endpoint that failed silently on Delivery
Radar's first attempt (see that tool's README for the full story). This
build uses a well-maintained library (`nse` on PyPI) that handles session
cookies more carefully than the first attempt did, which meaningfully
improves the odds — but it genuinely could not be tested against the live
site from a sandboxed environment. **If the first live run comes back with
zero deals, that's the same failure pattern as before, not a sign
something is broken in this code specifically** — check `errors` the same
way, and it's very fixable.

## The insider trading section, honestly

This is the least certain part of the product. NSE's general
announcements feed mixes every company disclosure together (results,
board meetings, litigation, insider trading, everything) with no clean
"insider trading only" filter available. This tool keyword-matches
against each announcement's own description to isolate likely insider
trading filings — it will miss some and occasionally include a false
positive. Treat it as a useful starting shortlist to click into, not a
complete, authoritative feed.

## Running it yourself

```bash
pip install -r requirements.txt

python scripts/scan.py --demo              # synthetic data, no network
python scripts/scan.py                     # live, last 10 calendar days
python scripts/scan.py --lookback-days 20  # look further back
```

## Reading the Symbol Summary

- **Net Value (Cr):** total disclosed buying minus total disclosed
  selling, in rupees crore. Green = net buying, red = net selling.
- **Repeat Buyers:** any client name that shows up as a BUY on that
  symbol 2+ times in the window — a real, deliberate build rather than a
  single opportunistic trade.
- Rows highlighted with an amber left edge are symbols on your watchlist.

## Where this fits with the other two

Same shape again: fetch → compute → JSON → static dashboard. The natural
next step once this is running for a while: a stock showing up here
*and* on Delivery Radar's quiet-accumulation list *and* approaching a
golden cross on DMA Radar is a meaningfully stronger shortlist than any
one signal alone.
