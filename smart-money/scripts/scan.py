#!/usr/bin/env python3
"""
Smart Money Feed — Bulk/Block Deals + Insider Trading  (v3)
--------------------------------------------------------------
v1 -> v2 -> v3 history, so future-you knows why things look the way they do:
  v1: bulk/block deals + best-effort insider trading disclosures.
  v2: + known-investor watchlist, investor-type tagging, short-selling.
  v3: + a persistent local cache (deals_cache.csv, short_sell_cache.csv)
      so history accumulates across runs instead of re-fetching the same
      window every day, and + price follow-through for known-investor
      deals (did the stock actually move after they bought?).

DATA SOURCE HONESTY NOTE: this product talks to NSE's interactive API via
the `nse` PyPI package (session cookies cached to disk, server=True mode
built for unattended use) - the same category of endpoint that failed
silently on Delivery Radar's first attempt, mitigated but not eliminated.
Price follow-through uses Yahoo Finance, same as DMA Radar - proven
reliable in this project already.

CACHE DESIGN: only raw facts are cached (date, symbol, client, qty, price
etc.) - NOT derived fields like investor_type or is_known_investor. Those
are recomputed fresh from the CURRENT known_investors.txt on every run,
against the FULL cached history. Practical effect: add a new name to
known_investors.txt, and their entire cached trading history gets tagged
retroactively on the next run - not just new deals going forward.

First run backfills up to 365 days (NSE's own per-request cap on
bulkdeals()). Every run after that only fetches the gap since the last
cached date (plus a small safety overlap), so the cache grows over many
days into a multi-year history without ever re-asking NSE for data it
already has. Retention is capped (see CACHE_RETENTION_DAYS) so the file
doesn't grow forever.

Usage:
    python scan.py --demo              # synthetic data, no network calls
    python scan.py                     # live, incremental cache update
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("smart-money")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
DEFAULT_KNOWN_INVESTORS = ROOT / "known_investors.txt"
OUTPUT_PATH = ROOT / "data" / "scan_results.json"
NSE_CACHE_DIR = ROOT / ".nse_cache"

DEALS_CACHE_PATH = ROOT / "data" / "deals_cache.csv"
SHORT_CACHE_PATH = ROOT / "data" / "short_sell_cache.csv"

NSE_MAX_RANGE_DAYS = 365        # NSE's own hard cap per bulkdeals() call
CACHE_RETENTION_DAYS = 730      # keep ~2 years, trim anything older
CATCHUP_BUFFER_DAYS = 5         # small overlap on incremental fetches, in case NSE data corrects late
RECENT_DEALS_DISPLAY_DAYS = 60  # "Recent Deals" table only needs to be recent, not the whole cache
MAX_FOLLOWTHROUGH_LOOKUPS = 40  # cap on price-history calls per run, keeps runtime bounded

DEALS_CACHE_COLUMNS = ["date", "symbol", "scrip_name", "client_name", "buy_sell", "qty", "price", "deal_type"]
SHORT_CACHE_COLUMNS = ["date", "symbol", "scrip_name", "qty"]

INSIDER_KEYWORDS = [
    "insider trading", "sast", "pit regulation", "regulation 29",
    "regulation 31", "encumbrance", "acquisition of shares", "promoter shareholding",
    "substantial acquisition",
]

INVESTOR_TYPE_PATTERNS = [
    ("Mutual Fund", ["MUTUAL FUND", "ASSET MANAGEMENT", " AMC "]),
    ("Insurance", ["INSURANCE", "LIC OF INDIA", "LIFE INSURANCE"]),
    ("Pension/PF", ["PENSION FUND", "PROVIDENT FUND", "GRATUITY FUND"]),
    ("FII/FPI", ["FPI", "FOREIGN PORTFOLIO", "PTE", " LLC", "MASTER FUND",
                 "OFFSHORE", "SICAV", "GMBH", "SINGAPORE", "MAURITIUS",
                 "LUXEMBOURG", "CAYMAN", "JERSEY", "FUND"]),
    ("HUF/Trust", [" HUF", "TRUST"]),
    ("Corporate/LLP", ["PRIVATE LIMITED", "PVT LTD", "PVT. LTD", " LIMITED",
                        " LLP", "CORPORATION", "ENTERPRISES"]),
]


# ---------------------------------------------------------------------------
# Watchlists / config files
# ---------------------------------------------------------------------------

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
    name_upper = client_name.upper()
    for known in known_investors:
        if known in name_upper:
            return known
    return None


def classify_investor_type(client_name: str) -> str:
    name_upper = f" {client_name.upper()} "
    for label, keywords in INVESTOR_TYPE_PATTERNS:
        if any(kw in name_upper for kw in keywords):
            return label
    return "Individual/HNI"


# ---------------------------------------------------------------------------
# Generic CSV cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return pd.DataFrame(columns=columns)


def save_cache(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["date", "symbol"]).to_csv(path, index=False)


def determine_fetch_window(cache: pd.DataFrame) -> tuple[datetime, datetime]:
    """Returns (from_date, to_date) to request from NSE. First run: max
    allowed range. Later runs: gap since most recent cached date, plus a
    small safety buffer. Deliberately tz-naive throughout - these are
    calendar dates, and the cache itself is stored tz-naive."""
    to_date = datetime.now(timezone.utc).replace(tzinfo=None)
    if cache.empty:
        from_date = to_date - timedelta(days=NSE_MAX_RANGE_DAYS)
        log.info("No existing cache - backfilling the max %d days NSE allows per request.", NSE_MAX_RANGE_DAYS)
    else:
        most_recent = cache["date"].max()
        if hasattr(most_recent, "tzinfo") and most_recent.tzinfo is not None:
            most_recent = most_recent.tz_localize(None)
        from_date = most_recent.to_pydatetime() - timedelta(days=CATCHUP_BUFFER_DAYS)
        from_date = max(from_date, to_date - timedelta(days=NSE_MAX_RANGE_DAYS))  # never exceed NSE's cap
        log.info("Existing cache found through %s - fetching from %s.", most_recent.date(), from_date.date())
    return from_date, to_date


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


def fetch_price_followthrough_live(symbol: str, deal_date: datetime) -> dict | None:
    import yfinance as yf

    try:
        start = deal_date - timedelta(days=5)
        end = min(deal_date + timedelta(days=35), datetime.now(timezone.utc)) + timedelta(days=1)
        df = yf.Ticker(f"{symbol}.NS").history(start=start, end=end, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
        df.index = df.index.tz_localize(None)
        df = df.sort_index()

        deal_ts = pd.Timestamp(deal_date.date())
        on_or_after = df[df.index >= deal_ts]
        if on_or_after.empty:
            return None
        baseline_price = float(on_or_after.iloc[0]["Close"])
        baseline_date = on_or_after.index[0]

        result = {"baseline_price": round(baseline_price, 2), "baseline_date": baseline_date.strftime("%Y-%m-%d")}
        future = df[df.index > baseline_date]
        for label, n in (("return_5d", 5), ("return_10d", 10), ("return_20d", 20)):
            if len(future) >= n:
                fut_price = float(future.iloc[n - 1]["Close"])
                result[label] = round((fut_price - baseline_price) / baseline_price * 100, 2)
            else:
                result[label] = None  # not enough trading days have elapsed yet
        return result
    except Exception as exc:  # noqa: BLE001
        log.debug("Follow-through fetch failed for %s @ %s: %s", symbol, deal_date, exc)
        return None


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def demo_deals_window(option_type: str, from_date: datetime, to_date: datetime) -> list[dict]:
    rng = np.random.default_rng(1 if option_type == "bulk_deals" else 2)
    symbols = ["KAYNES", "DATAPATTNS", "ABB", "WABAG", "PARAS", "CONCORDBIO", "ROSSTECH", "SIGMAADV", "BLUEJET", "GARFIBRES"]
    clients = ["QUANT WEALTH ADVISORS LLP", "MORGAN STANLEY ASIA (SINGAPORE) PTE",
               "GOLDMAN SACHS FUNDS", "VIJAY KEDIA", "ASHISH KACHOLIA HUF",
               "NOMURA INDIA INVESTMENT FUND", "SBI MUTUAL FUND"]
    rows = []
    n_days = max((to_date - from_date).days, 1)
    for day_offset in range(n_days):
        if rng.random() > 0.55:
            continue
        d = to_date - timedelta(days=day_offset)
        for _ in range(int(rng.integers(1, 4))):
            rows.append({
                "BD_DT_DATE": d.strftime("%d-%b-%Y").upper(),
                "BD_SYMBOL": symbols[rng.integers(0, len(symbols))],
                "BD_SCRIP_NAME": "Demo Ltd",
                "BD_CLIENT_NAME": clients[rng.integers(0, len(clients))],
                "BD_BUY_SELL": "BUY" if rng.random() > 0.35 else "SELL",
                "BD_QTY_TRD": int(rng.integers(50_000, 2_000_000)),
                "BD_TP_WATP": round(float(rng.uniform(80, 3500)), 2),
            })
    if n_days > 10:
        for day_offset in (1, 3, 6, 15, 40):
            if day_offset >= n_days:
                continue
            rows.append({
                "BD_DT_DATE": (to_date - timedelta(days=day_offset)).strftime("%d-%b-%Y").upper(),
                "BD_SYMBOL": "PARAS", "BD_SCRIP_NAME": "Demo Ltd",
                "BD_CLIENT_NAME": "ASHISH KACHOLIA HUF", "BD_BUY_SELL": "BUY",
                "BD_QTY_TRD": int(rng.integers(100_000, 500_000)),
                "BD_TP_WATP": round(float(rng.uniform(80, 3500)), 2),
            })
    return rows


def demo_short_sell_window(from_date: datetime, to_date: datetime) -> list[dict]:
    rng = np.random.default_rng(4)
    symbols = ["KAYNES", "DATAPATTNS", "ABB", "WABAG", "PARAS", "CONCORDBIO", "ROSSTECH"]
    rows = []
    n_days = max((to_date - from_date).days, 1)
    for day_offset in range(n_days):
        if rng.random() > 0.45:
            continue
        d = to_date - timedelta(days=day_offset)
        for _ in range(int(rng.integers(1, 3))):
            rows.append({
                "SS_DATE": d.strftime("%d-%b-%Y").upper(),
                "SS_SYMBOL": symbols[rng.integers(0, len(symbols))], "SS_NAME": "Demo Ltd",
                "SS_QTY": int(rng.integers(500, 80_000)),
            })
    return rows


def demo_announcements(lookback_days: int) -> list[dict]:
    rng = np.random.default_rng(3)
    d = datetime.now(timezone.utc) - timedelta(days=int(rng.integers(0, max(lookback_days, 1))))
    return [{
        "symbol": "SIGMAADV", "sm_name": "Demo Ltd",
        "desc": "Disclosure under Regulation 29(2) - Insider Trading",
        "an_dt": d.strftime("%d-%b-%Y %H:%M:%S"),
        "attchmntText": "Promoter has acquired shares under SAST disclosure norms.",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/sample_filing.pdf",
    }]


def demo_followthrough(seed_key: str) -> dict:
    rng = np.random.default_rng(abs(hash(seed_key)) % (2**32))
    base = round(float(rng.uniform(80, 3500)), 2)
    return {
        "baseline_price": base, "baseline_date": "2026-05-01",
        "return_5d": round(float(rng.normal(1.5, 4)), 2),
        "return_10d": round(float(rng.normal(3, 7)), 2),
        "return_20d": round(float(rng.normal(5, 11)), 2),
    }


# ---------------------------------------------------------------------------
# Cache update (fetch only what's missing, merge, trim, save)
# ---------------------------------------------------------------------------

def normalize_deal_for_cache(raw: dict, deal_type: str) -> dict | None:
    try:
        return {
            "date": pd.to_datetime(raw["BD_DT_DATE"], format="%d-%b-%Y", errors="coerce"),
            "symbol": str(raw["BD_SYMBOL"]).strip().upper(),
            "scrip_name": raw.get("BD_SCRIP_NAME", ""),
            "client_name": str(raw.get("BD_CLIENT_NAME", "")).strip(),
            "buy_sell": raw.get("BD_BUY_SELL", "?"),
            "qty": int(raw["BD_QTY_TRD"]),
            "price": float(raw["BD_TP_WATP"]),
            "deal_type": deal_type,
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("Skipping malformed %s record: %s", deal_type, exc)
        return None


def normalize_short_for_cache(raw: dict) -> dict | None:
    try:
        return {
            "date": pd.to_datetime(raw["SS_DATE"], format="%d-%b-%Y", errors="coerce"),
            "symbol": str(raw["SS_SYMBOL"]).strip().upper(),
            "scrip_name": raw.get("SS_NAME", ""),
            "qty": int(raw["SS_QTY"]),
        }
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("Skipping malformed short-sell record: %s", exc)
        return None


def update_deals_cache(demo: bool) -> pd.DataFrame:
    cache = load_cache(DEALS_CACHE_PATH, DEALS_CACHE_COLUMNS)
    from_date, to_date = determine_fetch_window(cache)

    new_rows = []
    for option_type, label in (("bulk_deals", "bulk"), ("block_deals", "block")):
        raw = demo_deals_window(option_type, from_date, to_date) if demo else fetch_deals_live(option_type, from_date, to_date)
        log.info("%s: %d raw record(s) fetched for this window", label, len(raw))
        for r in raw:
            norm = normalize_deal_for_cache(r, label)
            if norm and pd.notna(norm["date"]):
                new_rows.append(norm)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=DEALS_CACHE_COLUMNS, keep="last")

    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=CACHE_RETENTION_DAYS))
    cache = cache[cache["date"] >= cutoff]
    save_cache(cache, DEALS_CACHE_PATH)
    log.info("Deals cache now holds %d rows spanning %s to %s",
              len(cache), cache["date"].min().date() if not cache.empty else "-",
              cache["date"].max().date() if not cache.empty else "-")
    return cache


def update_short_cache(demo: bool) -> pd.DataFrame:
    cache = load_cache(SHORT_CACHE_PATH, SHORT_CACHE_COLUMNS)
    from_date, to_date = determine_fetch_window(cache)

    raw = demo_short_sell_window(from_date, to_date) if demo else fetch_deals_live("short_selling", from_date, to_date)
    log.info("short_selling: %d raw record(s) fetched for this window", len(raw))
    new_rows = [n for r in raw if (n := normalize_short_for_cache(r)) is not None and pd.notna(n["date"])]

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=SHORT_CACHE_COLUMNS, keep="last")

    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=CACHE_RETENTION_DAYS))
    cache = cache[cache["date"] >= cutoff]
    save_cache(cache, SHORT_CACHE_PATH)
    log.info("Short-sell cache now holds %d rows", len(cache))
    return cache


# ---------------------------------------------------------------------------
# Processing (derived fields computed fresh from cache every run)
# ---------------------------------------------------------------------------

def enrich_deal(row: pd.Series, watch_set: set[str], known_investors: list[str]) -> dict:
    client_name = row["client_name"] if pd.notna(row["client_name"]) else ""
    matched_investor = match_known_investor(client_name, known_investors) if client_name else None
    qty, price = float(row["qty"]), float(row["price"])
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "symbol": row["symbol"],
        "scrip_name": row["scrip_name"] if pd.notna(row["scrip_name"]) else row["symbol"],
        "client_name": client_name,
        "investor_type": classify_investor_type(client_name) if client_name else "Unknown",
        "is_known_investor": matched_investor is not None,
        "matched_investor": matched_investor,
        "buy_sell": row["buy_sell"],
        "qty": int(qty),
        "price": round(price, 2),
        "value_cr": round(qty * price / 1e7, 3),
        "deal_type": row["deal_type"],
        "on_watchlist": row["symbol"] in watch_set if watch_set else None,
    }


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
            s["buy_qty"] += d["qty"]; s["buy_value_cr"] += d["value_cr"]
        elif d["buy_sell"] == "SELL":
            s["sell_qty"] += d["qty"]; s["sell_value_cr"] += d["value_cr"]
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
            "symbol": s["symbol"], "scrip_name": s["scrip_name"], "on_watchlist": s["on_watchlist"],
            "deal_count": s["deal_count"], "known_investors_involved": sorted(s["known_investors_involved"]),
            "net_qty": s["buy_qty"] - s["sell_qty"], "net_value_cr": round(s["buy_value_cr"] - s["sell_value_cr"], 3),
            "buy_value_cr": round(s["buy_value_cr"], 3), "sell_value_cr": round(s["sell_value_cr"], 3),
            "distinct_clients": len(s["clients"]), "repeat_buyers": repeat_buyers,
        })
    summary.sort(key=lambda r: (len(r["known_investors_involved"]) > 0, abs(r["net_value_cr"])), reverse=True)
    return summary


def build_short_sell_summary(short_cache: pd.DataFrame) -> list[dict]:
    if short_cache.empty:
        return []
    grouped = short_cache.groupby("symbol").agg(
        scrip_name=("scrip_name", "first"), total_qty=("qty", "sum"), days_shorted=("date", "nunique")
    ).reset_index()
    out = grouped.to_dict("records")
    out.sort(key=lambda r: r["total_qty"], reverse=True)
    return out


def attach_short_sell_context(symbol_summary: list[dict], short_sell_summary: list[dict]) -> None:
    short_by_symbol = {s["symbol"]: s for s in short_sell_summary}
    for row in symbol_summary:
        ss = short_by_symbol.get(row["symbol"])
        row["short_qty_total"] = int(ss["total_qty"]) if ss else 0
        row["short_days"] = int(ss["days_shorted"]) if ss else 0


def filter_insider_announcements(raw_announcements: list[dict]) -> list[dict]:
    out = []
    for a in raw_announcements:
        text = f"{a.get('desc', '')} {a.get('attchmntText', '')}".lower()
        if any(kw in text for kw in INSIDER_KEYWORDS):
            out.append({
                "date": a.get("an_dt", ""), "symbol": a.get("symbol", ""), "company": a.get("sm_name", ""),
                "description": a.get("desc", ""), "summary": a.get("attchmntText", ""),
                "filing_url": a.get("attchmntFile", ""),
            })
    return out


def add_price_followthrough(known_investor_deals: list[dict], demo: bool) -> list[dict]:
    """Attaches follow-through price data to known-investor deals, deduped
    by (symbol, date) since multiple investors sometimes trade the same
    name the same day. Capped at MAX_FOLLOWTHROUGH_LOOKUPS to keep runtime
    bounded regardless of how large the cache grows over time."""
    seen: dict[tuple, dict] = {}
    lookups_done = 0
    for d in known_investor_deals:
        key = (d["symbol"], d["date"])
        if key in seen:
            d["followthrough"] = seen[key]
            continue
        if lookups_done >= MAX_FOLLOWTHROUGH_LOOKUPS:
            d["followthrough"] = None
            continue
        if demo:
            ft = demo_followthrough(f"{d['symbol']}{d['date']}")
        else:
            deal_dt = datetime.strptime(d["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            ft = fetch_price_followthrough_live(d["symbol"], deal_dt)
        seen[key] = ft
        d["followthrough"] = ft
        lookups_done += 1
    return known_investor_deals


def summarize_followthrough_by_investor(known_investor_deals: list[dict]) -> list[dict]:
    by_investor: dict[str, dict] = {}
    for d in known_investor_deals:
        ft = d.get("followthrough")
        if not ft or ft.get("return_20d") is None:
            continue
        inv = d["matched_investor"]
        s = by_investor.setdefault(inv, {"investor": inv, "returns_20d": [], "wins": 0, "trades": 0})
        s["trades"] += 1
        s["returns_20d"].append(ft["return_20d"])
        if ft["return_20d"] > 0:
            s["wins"] += 1
    out = []
    for s in by_investor.values():
        avg_return = sum(s["returns_20d"]) / len(s["returns_20d"])
        out.append({
            "investor": s["investor"], "trades_with_data": s["trades"],
            "avg_return_20d": round(avg_return, 2), "hit_rate_20d": round(s["wins"] / s["trades"] * 100, 1),
        })
    out.sort(key=lambda r: r["avg_return_20d"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_technical_context() -> dict[str, dict]:
    """Reads DMA Radar's and Delivery Radar's own latest output (sibling
    file reads, zero extra network calls) - completes the loop. DMA and
    Delivery already read Smart Money's data; without this, the
    interconnection would only run one direction. A known-investor deal
    on a stock that's ALSO showing a technical golden cross or quiet
    accumulation is a materially different situation than one sitting in
    isolation, and there was previously no way to see that from this
    dashboard alone."""
    context = {}
    dma_path = ROOT.parent / "dma-radar" / "data" / "scan_results.json"
    if dma_path.exists():
        try:
            for r in json.loads(dma_path.read_text()).get("results", []):
                context.setdefault(r["symbol"], {})["dma_signal"] = r.get("signal")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read dma-radar output: %s", exc)
    delivery_path = ROOT.parent / "delivery-radar" / "data" / "scan_results.json"
    if delivery_path.exists():
        try:
            for r in json.loads(delivery_path.read_text()).get("results", []):
                context.setdefault(r["symbol"], {})["delivery_signal"] = r.get("signal")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read delivery-radar output: %s", exc)
    return context


def main():
    parser = argparse.ArgumentParser(description="Smart Money Feed v3: bulk/block/short deals + insider trading, cached + follow-through")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--known-investors", type=Path, default=DEFAULT_KNOWN_INVESTORS)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    watch_set = load_watchlist(args.watchlist)
    known_investors = load_known_investors(args.known_investors)
    log.info("Watchlist: %d symbols, %d known investors (demo=%s)", len(watch_set), len(known_investors), args.demo)

    deals_cache = update_deals_cache(demo=args.demo)
    short_cache = update_short_cache(demo=args.demo)

    all_deals = [enrich_deal(row, watch_set, known_investors) for _, row in deals_cache.iterrows()]
    all_deals.sort(key=lambda d: d["date"], reverse=True)

    symbol_summary = build_symbol_summary(all_deals)
    short_sell_summary = build_short_sell_summary(short_cache)
    attach_short_sell_context(symbol_summary, short_sell_summary)

    technical_context = load_technical_context()
    NO_SIG = ("no_signal", None)
    for row in symbol_summary:
        ctx = technical_context.get(row["symbol"], {})
        row["dma_signal"] = ctx.get("dma_signal") if ctx.get("dma_signal") not in NO_SIG else None
        row["delivery_signal"] = ctx.get("delivery_signal") if ctx.get("delivery_signal") not in NO_SIG else None
        row["technical_alignment_count"] = sum(1 for x in (row["dma_signal"], row["delivery_signal"]) if x)
    # Re-sort now that technical alignment is known: known-investor presence
    # still wins first, but among ties, a stock corroborated by DMA and/or
    # Delivery Radar too now ranks above one with no technical confirmation.
    symbol_summary.sort(
        key=lambda r: (len(r["known_investors_involved"]) > 0, r["technical_alignment_count"], abs(r["net_value_cr"])),
        reverse=True)

    recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_DEALS_DISPLAY_DAYS)).strftime("%Y-%m-%d")
    recent_deals = [d for d in all_deals if d["date"] >= recent_cutoff]

    known_investor_deals = [d for d in all_deals if d["is_known_investor"]]
    log.info("Known-investor deals in full cache: %d out of %d total", len(known_investor_deals), len(all_deals))
    known_investor_deals = add_price_followthrough(known_investor_deals, demo=args.demo)
    followthrough_by_investor = summarize_followthrough_by_investor(known_investor_deals)

    investor_type_breakdown: dict[str, int] = {}
    for d in all_deals:
        investor_type_breakdown[d["investor_type"]] = investor_type_breakdown.get(d["investor_type"], 0) + 1

    lookback_days_effective = (pd.Timestamp(datetime.now(timezone.utc).date()) - deals_cache["date"].min()).days if not deals_cache.empty else 0

    if args.demo:
        raw_announcements = demo_announcements(30)
    else:
        raw_announcements = fetch_announcements_live(datetime.now(timezone.utc) - timedelta(days=30), datetime.now(timezone.utc))
    insider_filings = filter_insider_announcements(raw_announcements)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_span_days": lookback_days_effective,
        "recent_display_days": RECENT_DEALS_DISPLAY_DAYS,
        "watchlist_size": len(watch_set),
        "known_investors_tracked": len(known_investors),
        "deal_count_total_cached": len(all_deals),
        "deal_count_recent": len(recent_deals),
        "deals": recent_deals,
        "symbol_summary": symbol_summary,
        "known_investor_deals": known_investor_deals[:150],  # cap JSON size; still full history in the cache file
        "followthrough_by_investor": followthrough_by_investor,
        "investor_type_breakdown": investor_type_breakdown,
        "short_sell_summary": short_sell_summary,
        "insider_trading": {
            "note": "Best-effort keyword match against NSE's general announcements feed - "
                    "not a guaranteed-complete list.",
            "announcements_scanned": len(raw_announcements),
            "filings": insider_filings,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d cached deals, %d known-investor, %d w/ follow-through data, %d insider filings)",
              args.out, len(all_deals), len(known_investor_deals),
              sum(1 for d in known_investor_deals if d.get("followthrough")), len(insider_filings))


if __name__ == "__main__":
    main()
