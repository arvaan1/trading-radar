#!/usr/bin/env python3
"""
Universe builder — expands watchlist.txt from ~50 stocks to ~500-1000.
------------------------------------------------------------------------
Run this manually (via the "Update Universe" GitHub Action, or locally)
whenever you want to refresh the stock universe. It is NOT part of the
daily scan workflows on purpose: NSE's index membership only changes a
few times a year, so re-fetching it daily would just be extra unnecessary
load against NSE for no benefit. Run it now, then again every few months.

What it does:
    1. Downloads NSE's own official constituent lists for:
       - NIFTY 500 (large + mid + small cap, ~500 names)
       - NIFTY Microcap 250 (the next 250 below that, ~250 names)
    2. Adds every symbol from universe_extras.txt (hand-picked names -
       useful for very recent listings or SME-platform stocks that
       aren't index-eligible yet).
    3. Deduplicates, sorts, and writes the combined list to BOTH
       dma-radar/watchlist.txt and delivery-radar/watchlist.txt.

Why pull from NSE's indices instead of a hand-typed list: every symbol
here already cleared NSE's own liquidity/eligibility bar for index
inclusion - a real quality filter, not just "stocks someone typed in."

HONESTY NOTE: the exact NSE Microcap 250 CSV filename could not be
verified from a fully sandboxed environment, so this script tries a
couple of likely variants and reports clearly which one (if any)
worked. If a run doesn't get Microcap 250, you still end up with the
Nifty 500 + your extras (500-550+ names) - not a total failure, just a
smaller win than the full pull. Check the run's log output either way.
"""

import io
import sys
import logging
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("universe-builder")

ROOT = Path(__file__).resolve().parent
EXTRAS_PATH = ROOT / "universe_extras.txt"
TARGET_PATHS = [ROOT / "dma-radar" / "watchlist.txt", ROOT / "delivery-radar" / "watchlist.txt"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

# Each index tries these candidate URLs in order, first one that parses wins.
INDEX_SOURCES = {
    "NIFTY 500": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ],
    "NIFTY MICROCAP 250": [
        "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
    ],
}


def _normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_symbol_column(columns) -> str | None:
    for c in columns:
        if _normalize(c) == "symbol":
            return c
    return None


def fetch_index_symbols(index_name: str, urls: list[str]) -> set[str]:
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code != 200 or not resp.text.strip():
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            sym_col = find_symbol_column(df.columns)
            if sym_col is None:
                continue
            symbols = set(df[sym_col].astype(str).str.strip().str.upper())
            symbols.discard("")
            if symbols:
                log.info("%s: got %d symbols from %s", index_name, len(symbols), url)
                return symbols
        except Exception as exc:  # noqa: BLE001
            log.debug("%s: %s failed (%s)", index_name, url, exc)
            continue
    log.warning("%s: could not fetch from any candidate URL - skipping this index.", index_name)
    return set()


def load_extras(path: Path) -> set[str]:
    if not path.exists():
        log.warning("No extras file at %s - skipping.", path)
        return set()
    symbols = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip().upper()  # strip inline comments
        if line:
            symbols.add(line)
    log.info("Loaded %d hand-picked extras from %s", len(symbols), path.name)
    return symbols


def main():
    all_symbols = set()
    for index_name, urls in INDEX_SOURCES.items():
        all_symbols |= fetch_index_symbols(index_name, urls)

    extras = load_extras(EXTRAS_PATH)
    all_symbols |= extras

    if not all_symbols:
        log.error("Ended up with zero symbols - not overwriting existing watchlists. Aborting.")
        sys.exit(1)

    final_list = sorted(all_symbols)
    log.info("Final combined universe: %d unique symbols.", len(final_list))

    header = (
        "# Auto-generated by build_universe.py from NSE's Nifty 500 + Nifty Microcap 250\n"
        "# index lists, plus universe_extras.txt. Do not hand-edit this file directly -\n"
        "# edit universe_extras.txt for permanent additions, or re-run the Update Universe\n"
        "# workflow to refresh from NSE. Manual edits here will be overwritten next run.\n"
    )

    for target in TARGET_PATHS:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header + "\n".join(final_list) + "\n")
        log.info("Wrote %d symbols to %s", len(final_list), target)


if __name__ == "__main__":
    main()
