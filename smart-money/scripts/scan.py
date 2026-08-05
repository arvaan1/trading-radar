#!/usr/bin/env python3
"""
Smart Money Feed — Bulk/Block Deals + Insider Trading
--------------------------------------------------------
Pulls NSE's daily bulk deal and block deal disclosures (large trades by
institutions/HNIs, publicly disclosed same-day) and - best effort -
insider trading disclosures, then aggregates by symbol so repeat activity
in the same name stands out instead of getting lost in a long raw list.

DATA SOURCE HONESTY NOTE: unlike DMA Radar (Yahoo Finance) and Delivery
Radar (NSE's plain daily file download), this product talks to NSE's
interactive API - the same category of endpoint that failed silently on
Delivery Radar's first attempt. This build uses the `nse` PyPI package
instead of hand-rolled requests, because it caches session cookies to
disk (far fewer handshakes than a fresh session per call) and has a
`server=True` mode built specifically for unattended/script use rather
than an interactive browser session - the best available mitigation for
the cloud-IP bot-detection issue we hit before. It could not be tested
against the live site from this sandboxed environment. If the bulk/block
section comes back empty on first live run, that's the same failure
pattern as before and worth reporting back rather than assuming it's a
bug in this script.

The insider-trading section is the least certain part of this product:
NSE's general corporate-announcements feed does not cleanly separate
"insider trading disclosure" from every other company announcement
(results, board meetings, litigation, etc.), so this filters on keyword
matches against the announcement's own description field. Treat this
section as a best-effort start, not a guaranteed-complete feed - it may
need retuning once you see real output.

Usage:
    python scan.py --demo              # synthetic data, no network calls
    python scan.py                     # live: last 10 calendar days
    python scan.py --lookback-days 20
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("smart-money")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
DEFAULT_KNOWN_INVESTORS = ROOT / "known_investors.txt"
OUTPUT_PATH = ROOT / "data" / "scan_results.json"
NSE_CACHE_DIR = ROOT / ".nse_cache"

# Keywords used to best-effort isolate insider-trading disclosures from
# NSE's general corporate-announcements feed. Deliberately broad (SEBI's
# PIT Regulations disclosure requirements are usually filed under
# Regulation 29/30/31 language) - a false positive here just shows an
# unrelated announcement, which is far less costly than missing a real one.
INSIDER_KEYWORDS = [
    "insider trading", "sast", "pit regulation", "regulation 29",
    "regulation 31", "encumbrance", "acquisition of shares", "promoter shareholding",
    "substantial acquisition",
]


def load_watchlist(path: Path) -> set[str]:
    if not path.exists():
        log.warning("Watchlist not found at %s - watchlist-highlighting will be off.", path)
        return set()
    return {
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_known_investors(path: Path) -> list[str]:
    if not path.exists():
        log.warning("No known_investors.txt found at %s - known-investor tagging will be off.", path)
        return []
    names = [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    log.info("Loaded %d known investor name(s) to watch for.", len(names))
    return names


def match_known_investor(client_name: str, known_investors: list[str]) -> str | None:
    """Case-insensitive substring match. Returns the matched watchlist
    entry (not the raw client name) so the dashboard can group by a
    consistent label even when NSE's formatting varies."""
    name_upper = client_name.upper()
    for known in known_investors:
        if known in name_upper:
            return known
    return None


# Coarse investor-type classification from client-name patterns. This is
# a heuristic, not an authoritative registry lookup - good enough to spot
# "is this a fund or a person" at a glance, not guaranteed precise.
INVESTOR_TYPE_PATTERNS = [
    ("Mutual Fund", ["MUTUAL FUND", "ASSET MANAGEMENT", " AMC "]),
    ("Insurance", ["INSURANCE", "LIC OF INDIA", "LIFE INSURANCE"]),
    ("Pension/PF", ["PENSION FUND", "PROVIDENT FUND", "GRATUITY FUND"]),
    # Broad net for foreign institutional naming: explicit FPI language,
    # common offshore jurisdictions, or a generic "___ FUND" that isn't
    # one of the specific domestic fund types already matched above.
    ("FII/FPI", ["FPI", "FOREIGN PORTFOLIO", "PTE", " LLC", "MASTER FUND",
                 "OFFSHORE", "SICAV", "GMBH", "SINGAPORE", "MAURITIUS",
                 "LUXEMBOURG", "CAYMAN", "JERSEY", "FUND"]),
    ("HUF/Trust", [" HUF", "TRUST"]),
    ("Corporate/LLP", ["PRIVATE LIMITED", "PVT LTD", "PVT. LTD", " LIMITED",
                        " LLP", "CORPORATION", "ENTERPRISES"]),
]


def classify_investor_type(client_name: str) -> str:
    name_upper = f" {client_name.upper()} "
    for label, keywords in INVESTOR_TYPE_PATTERNS:
        if any(kw in name_upper for kw in keywords):
            return label
    return "Individual/HNI"  # sensible default - most unmatched names are plain personal names


# ---------------------------------------------------------------------------
# Live data fetch
# ---------------------------------------------------------------------------

def fetch_deals_live(option_type: str, from_date: datetime, to_date: datetime) -> list[dict]:
    from nse import NSE

    NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with NSE(str(NSE_CACHE_DIR), server=True) as nse:
        try:
            return nse.bulkdeals(option_type, from_date, to_date)
        except RuntimeError as exc:
            log.warning("%s: %s", option_type, exc)
            return []


def fetch_announcements_live(from_date: datetime, to_date: datetime) -> list[dict]:
    from nse import NSE

    NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with NSE(str(NSE_CACHE_DIR), server=True) as nse:
        try:
            return nse.announcements(from_date=from_date, to_date=to_date)
        except Exception as exc:  # noqa: BLE001
            log.warning("announcements fetch failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def demo_deals(option_type: str, watch_symbols: list[str], lookback_days: int) -> list[dict]:
    rng = np.random.default_rng(1 if option_type == "bulk_deals" else 2)
    clients = ["QUANT WEALTH ADVISORS LLP", "MORGAN STANLEY ASIA (SINGAPORE) PTE",
               "GOLDMAN SACHS FUNDS", "VIJAY KEDIA", "ASHISH KACHOLIA HUF",
               "NOMURA INDIA INVESTMENT FUND", "SBI MUTUAL FUND"]
    rows = []
    for day_offset in range(lookback_days):
        if rng.random() > 0.55:  # not every day has deals
            continue
        n_deals = rng.integers(1, 4)
        d = datetime.now(timezone.utc) - timedelta(days=day_offset)
        for _ in range(n_deals):
            sym = watch_symbols[rng.integers(0, len(watch_symbols))]
            client = clients[rng.integers(0, len(clients))]
            rows.append({
                "BD_DT_DATE": d.strftime("%d-%b-%Y").upper(),
                "BD_SYMBOL": sym,
                "BD_SCRIP_NAME": f"{sym} Limited",
                "BD_CLIENT_NAME": client,
                "BD_BUY_SELL": "BUY" if rng.random() > 0.35 else "SELL",
                "BD_QTY_TRD": int(rng.integers(50_000, 2_000_000)),
                "BD_TP_WATP": round(float(rng.uniform(80, 3500)), 2),
            })
    # inject one clear repeat-buyer pattern so the signal is visible in demo output
    if watch_symbols:
        target_sym, target_client = watch_symbols[0], "ASHISH KACHOLIA HUF"
        for day_offset in (1, 3, 6):
            rows.append({
                "BD_DT_DATE": (datetime.now(timezone.utc) - timedelta(days=day_offset)).strftime("%d-%b-%Y").upper(),
                "BD_SYMBOL": target_sym, "BD_SCRIP_NAME": f"{target_sym} Limited",
                "BD_CLIENT_NAME": target_client, "BD_BUY_SELL": "BUY",
                "BD_QTY_TRD": int(rng.integers(100_000, 500_000)),
                "BD_TP_WATP": round(float(rng.uniform(80, 3500)), 2),
            })
    return rows


def demo_announcements(watch_symbols: list[str], lookback_days: int) -> list[dict]:
    rng = np.random.default_rng(3)
    if not watch_symbols:
        return []
    sym = watch_symbols[rng.integers(0, len(watch_symbols))]
    d = datetime.now(timezone.utc) - timedelta(days=int(rng.integers(0, lookback_days)))
    return [{
        "symbol": sym, "sm_name": f"{sym} Limited",
        "desc": "Disclosure under Regulation 29(2) - Insider Trading",
        "an_dt": d.strftime("%d-%b-%Y %H:%M:%S"),
        "attchmntText": f"Promoter of {sym} Limited has acquired shares under SAST disclosure norms.",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/sample_filing.pdf",
    }]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def normalize_deal(raw: dict, deal_type: str, watch_set: set[str], known_investors: list[str]) -> dict | None:
    try:
        symbol = str(raw["BD_SYMBOL"]).strip().upper()
        qty = int(raw["BD_QTY_TRD"])
        price = float(raw["BD_TP_WATP"])
        client_name = str(raw.get("BD_CLIENT_NAME", "")).strip()
        matched_investor = match_known_investor(client_name, known_investors) if client_name else None
        return {
            "date": raw["BD_DT_DATE"],
            "symbol": symbol,
            "scrip_name": raw.get("BD_SCRIP_NAME", symbol),
            "client_name": client_name,
            "investor_type": classify_investor_type(client_name) if client_name else "Unknown",
            "is_known_investor": matched_investor is not None,
            "matched_investor": matched_investor,
            "buy_sell": raw.get("BD_BUY_SELL", "?"),
            "qty": qty,
            "price": price,
            "value_cr": round(qty * price / 1e7, 3),
            "deal_type": deal_type,
            "on_watchlist": symbol in watch_set if watch_set else None,
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("Skipping malformed %s record: %s", deal_type, exc)
        return None


def build_symbol_summary(deals: list[dict]) -> list[dict]:
    by_symbol: dict[str, dict] = {}
    for d in deals:
        s = by_symbol.setdefault(d["symbol"], {
            "symbol": d["symbol"], "scrip_name": d["scrip_name"],
            "buy_qty": 0, "sell_qty": 0, "buy_value_cr": 0.0, "sell_value_cr": 0.0,
            "deal_count": 0, "clients": {}, "on_watchlist": d["on_watchlist"],
            "known_investors_involved": set(),
        })
        s["deal_count"] += 1
        if d["buy_sell"] == "BUY":
            s["buy_qty"] += d["qty"]
            s["buy_value_cr"] += d["value_cr"]
        elif d["buy_sell"] == "SELL":
            s["sell_qty"] += d["qty"]
            s["sell_value_cr"] += d["value_cr"]
        if d["client_name"]:
            s["clients"].setdefault(d["client_name"], {"buy": 0, "sell": 0})
            if d["buy_sell"] == "BUY":
                s["clients"][d["client_name"]]["buy"] += 1
            elif d["buy_sell"] == "SELL":
                s["clients"][d["client_name"]]["sell"] += 1
        if d["is_known_investor"]:
            s["known_investors_involved"].add(d["matched_investor"])

    summary = []
    for s in by_symbol.values():
        repeat_buyers = [c for c, counts in s["clients"].items() if counts["buy"] >= 2]
        summary.append({
            "symbol": s["symbol"],
            "scrip_name": s["scrip_name"],
            "on_watchlist": s["on_watchlist"],
            "deal_count": s["deal_count"],
            "known_investors_involved": sorted(s["known_investors_involved"]),
            "net_qty": s["buy_qty"] - s["sell_qty"],
            "net_value_cr": round(s["buy_value_cr"] - s["sell_value_cr"], 3),
            "buy_value_cr": round(s["buy_value_cr"], 3),
            "sell_value_cr": round(s["sell_value_cr"], 3),
            "distinct_clients": len(s["clients"]),
            "repeat_buyers": repeat_buyers,
        })
    summary.sort(key=lambda r: (len(r["known_investors_involved"]) > 0, abs(r["net_value_cr"])), reverse=True)
    return summary


def filter_insider_announcements(raw_announcements: list[dict]) -> list[dict]:
    out = []
    for a in raw_announcements:
        text = f"{a.get('desc', '')} {a.get('attchmntText', '')}".lower()
        if any(kw in text for kw in INSIDER_KEYWORDS):
            out.append({
                "date": a.get("an_dt", ""),
                "symbol": a.get("symbol", ""),
                "company": a.get("sm_name", ""),
                "description": a.get("desc", ""),
                "summary": a.get("attchmntText", ""),
                "filing_url": a.get("attchmntFile", ""),
            })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Money Feed: bulk/block deals + insider trading")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--known-investors", type=Path, default=DEFAULT_KNOWN_INVESTORS)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    watch_set = load_watchlist(args.watchlist)
    watch_list_for_demo = sorted(watch_set) if watch_set else ["RELIANCE", "TCS", "KAYNES"]
    known_investors = load_known_investors(args.known_investors)
    log.info("Watchlist: %d symbols (demo=%s)", len(watch_set), args.demo)

    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=args.lookback_days)

    all_deals_raw = []
    for option_type, label in (("bulk_deals", "bulk"), ("block_deals", "block")):
        if args.demo:
            raw = demo_deals(option_type, watch_list_for_demo, args.lookback_days)
        else:
            raw = fetch_deals_live(option_type, from_date, to_date)
        log.info("%s: %d raw record(s)", label, len(raw))
        for r in raw:
            norm = normalize_deal(r, label, watch_set, known_investors)
            if norm:
                all_deals_raw.append(norm)

    all_deals_raw.sort(key=lambda d: d["date"], reverse=True)
    symbol_summary = build_symbol_summary(all_deals_raw)

    known_investor_deals = [d for d in all_deals_raw if d["is_known_investor"]]
    log.info("Known-investor deals matched: %d out of %d total", len(known_investor_deals), len(all_deals_raw))

    investor_type_breakdown: dict[str, int] = {}
    for d in all_deals_raw:
        investor_type_breakdown[d["investor_type"]] = investor_type_breakdown.get(d["investor_type"], 0) + 1

    if args.demo:
        raw_announcements = demo_announcements(watch_list_for_demo, args.lookback_days)
    else:
        raw_announcements = fetch_announcements_live(from_date, to_date)
    insider_filings = filter_insider_announcements(raw_announcements)
    log.info("Insider trading (best-effort): %d matched out of %d announcements scanned",
              len(insider_filings), len(raw_announcements))

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.lookback_days,
        "watchlist_size": len(watch_set),
        "known_investors_tracked": len(known_investors),
        "deal_count": len(all_deals_raw),
        "deals": all_deals_raw,
        "symbol_summary": symbol_summary,
        "known_investor_deals": known_investor_deals,
        "investor_type_breakdown": investor_type_breakdown,
        "insider_trading": {
            "note": "Best-effort keyword match against NSE's general announcements feed - "
                    "not a guaranteed-complete list. See scan.py docstring.",
            "announcements_scanned": len(raw_announcements),
            "filings": insider_filings,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d deals, %d known-investor, %d insider filings)",
              args.out, len(all_deals_raw), len(known_investor_deals), len(insider_filings))


if __name__ == "__main__":
    main()
