#!/usr/bin/env python3
"""
Delivery % / Volume Anomaly Radar  (v2 - bhavcopy-based)
------------------------------------------------------------
Same signals as v1 (see README), different plumbing underneath.

WHY THIS VERSION EXISTS: v1 called NSE's per-symbol historical-data API,
which requires first fetching a "visitor pass" (cookies) from nseindia.com
before every request. That two-step handshake is exactly the pattern
financial sites use bot-detection on, and cloud CI runners (like GitHub
Actions) get flagged more than home connections - which showed up as scans
silently returning empty or partial data rather than a clear error.

This version instead downloads NSE's public daily "Bhavcopy" file - a
single CSV NSE publishes for EVERY listed stock, every trading day, at a
predictable URL. It's a plain file download, not an interactive API call,
so there's no cookie handshake to fail. One file covers your entire
watchlist, so it's also far fewer requests overall: ~1/day once the cache
is warm, instead of ~1-per-symbol-per-day.

Mechanics:
    - Maintains data/bhavcopy_cache.csv: a rolling ~100-calendar-day
      history of (date, symbol, close, prev_close, avg_price, traded_qty,
      deliverable_qty, delivery_pct) for every symbol in your watchlist.
    - Each run, only downloads whatever trading days are missing since the
      last run (usually just 1). First run backfills ~130 calendar days
      to get enough history for the 60-day baseline.
    - Weekends/holidays are skipped automatically (NSE simply doesn't
      publish a file that day - a 404 there is expected, not an error).
    - Signal math (z-scores, categories, scoring) is IDENTICAL to v1.

Usage:
    python scan.py --demo     # synthetic data, exercises all 4 signal types
    python scan.py            # live: updates the cache, then scans
"""

import argparse
import io
import json
import sys
import time
import random
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("delivery-radar")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
OUTPUT_PATH = ROOT / "data" / "scan_results.json"
CACHE_PATH = ROOT / "data" / "bhavcopy_cache.csv"
CACHE_SYMBOLS_MARKER = ROOT / "data" / "cache_universe_marker.txt"

MIN_ROWS_FOR_20D = 25
MIN_ROWS_FOR_60D = 65
BACKFILL_CALENDAR_DAYS = 115   # first run: enough to comfortably cover 65+ trading days,
                                # without drastically overshooting CACHE_RETENTION_DAYS
CACHE_RETENTION_DAYS = 100     # trim anything older than this to keep the file small
CATCHUP_BUFFER_DAYS = 7        # small safety margin when resuming from an existing cache

BHAVCOPY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

CACHE_COLUMNS = ["date", "symbol", "prev_close", "close", "avg_price", "traded_qty", "deliverable_qty", "delivery_pct"]


def _normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_col(columns, candidates: list[str]) -> str | None:
    norm_map = {c: _normalize(c) for c in columns}
    for cand in candidates:
        cand_n = _normalize(cand)
        for orig, n in norm_map.items():
            if cand_n in n:
                return orig
    return None


BHAVCOPY_COLUMN_CANDIDATES = {
    "symbol": ["symbol"],
    "series": ["series"],
    "date": ["date1", "date"],
    "prev_close": ["prevclose"],
    "close": ["closeprice"],
    "avg_price": ["avgprice"],
    "traded_qty": ["ttltrdqnty", "totaltradedquantity"],
    "deliverable_qty": ["delivqty", "deliverableqty"],
    "delivery_pct": ["delivper", "dlyqttotradedqty"],
}


# ---------------------------------------------------------------------------
# Bhavcopy download + cache maintenance
# ---------------------------------------------------------------------------

def bhavcopy_url(date: datetime) -> str:
    return f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date.strftime('%d%m%Y')}.csv"


def download_bhavcopy_live(date: datetime) -> pd.DataFrame | None:
    """Returns None on weekends/holidays (file doesn't exist) or any fetch
    problem - this is expected and handled silently, not an error."""
    try:
        resp = requests.get(bhavcopy_url(date), headers=BHAVCOPY_HEADERS, timeout=20)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
        return df
    except Exception as exc:  # noqa: BLE001
        log.debug("bhavcopy fetch failed for %s: %s", date.date(), exc)
        return None


_DEMO_PRICE_WALKS: dict[str, pd.Series] = {}


def _get_demo_price_walk(symbol: str) -> pd.Series:
    """A real, continuous random walk per symbol (seeded deterministically
    from the symbol name so results are reproducible), covering enough
    history that any date the backfill asks for has a real answer. Built
    once per symbol and cached, since download_bhavcopy_demo() is called
    once per DAY (covering all symbols) - without this, each day's price
    was being generated independently from a hash of the symbol name only,
    which produced a perfectly flat price series (confirmed while testing
    the new backtest feature: std dev of 0.0 across 90+ days). That was
    invisible before because no prior feature depended on price actually
    changing day to day - only delivery % varying, which was seeded
    separately by date and did vary correctly."""
    if symbol in _DEMO_PRICE_WALKS:
        return _DEMO_PRICE_WALKS[symbol]
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    n = 420
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    base_price = 80 + 400 * ((abs(hash((symbol, "px"))) % 100) / 100)
    drift = rng.normal(0.0003, 0.017, n)
    walk = base_price * np.exp(np.cumsum(drift))
    series = pd.Series(walk, index=dates)
    _DEMO_PRICE_WALKS[symbol] = series
    return series


def download_bhavcopy_demo(date: datetime, symbols: list[str], seed_offset: int) -> pd.DataFrame:
    """Synthetic single-day bhavcopy covering the whole watchlist, used by
    --demo. Delivery % is deterministic per (date, symbol) so a full
    backfill produces a coherent time series, and seeded to inject the
    same four signal scenarios as v1's demo mode, on the most recent day
    only. Price now comes from a real per-symbol random walk (see
    _get_demo_price_walk) instead of a date-independent constant."""
    rng = np.random.default_rng(abs(hash((date.date(), seed_offset))) % (2**32))
    rows = []
    for i, sym in enumerate(symbols):
        base_delivery = 30 + 25 * ((abs(hash(sym)) % 100) / 100)
        delivery_pct = float(np.clip(rng.normal(base_delivery, 6), 5, 95))

        walk = _get_demo_price_walk(sym)
        nearest_idx = walk.index.get_indexer([pd.Timestamp(date.date())], method="nearest")[0]
        close = float(walk.iloc[nearest_idx])
        prev_close = float(walk.iloc[max(nearest_idx - 1, 0)])

        traded_qty = rng.integers(80_000, 900_000)
        rows.append({
            "SYMBOL": sym, "SERIES": "EQ", "DATE1": date.strftime("%d-%b-%Y"),
            "PREV_CLOSE": prev_close, "CLOSE_PRICE": close, "AVG_PRICE": close,
            "TTL_TRD_QNTY": traded_qty, "DELIV_QTY": traded_qty * delivery_pct / 100,
            "DELIV_PER": delivery_pct,
        })
    return pd.DataFrame(rows)


def normalize_bhavcopy(raw: pd.DataFrame, watch_symbols: set[str]) -> pd.DataFrame | None:
    cols = {k: find_col(raw.columns, v) for k, v in BHAVCOPY_COLUMN_CANDIDATES.items()}
    required = ("symbol", "series", "date", "prev_close", "close", "traded_qty", "deliverable_qty", "delivery_pct")
    if any(cols[k] is None for k in required):
        return None

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[cols["date"]], errors="coerce", dayfirst=True),
        "symbol": raw[cols["symbol"]].astype(str).str.strip().str.upper(),
        "series": raw[cols["series"]].astype(str).str.strip().str.upper(),
        "prev_close": pd.to_numeric(raw[cols["prev_close"]], errors="coerce"),
        "close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
        "avg_price": pd.to_numeric(raw[cols["avg_price"]], errors="coerce") if cols.get("avg_price") else np.nan,
        "traded_qty": pd.to_numeric(raw[cols["traded_qty"]], errors="coerce"),
        "deliverable_qty": pd.to_numeric(raw[cols["deliverable_qty"]], errors="coerce"),
        "delivery_pct": pd.to_numeric(raw[cols["delivery_pct"]], errors="coerce"),
    })
    df = df[(df["series"] == "EQ") & (df["symbol"].isin(watch_symbols))]
    df = df.dropna(subset=["date", "close", "traded_qty", "delivery_pct"])
    return df[CACHE_COLUMNS] if not df.empty else None


def load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        return df
    return pd.DataFrame(columns=CACHE_COLUMNS)


def save_cache(df: pd.DataFrame):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["symbol", "date"]).to_csv(CACHE_PATH, index=False)


def load_cache_marker() -> set[str]:
    """The set of symbols the cache was last built to cover. Used to detect
    when the watchlist has grown, so new symbols get properly backfilled
    instead of silently having near-zero history forever."""
    if CACHE_SYMBOLS_MARKER.exists():
        return set(s for s in CACHE_SYMBOLS_MARKER.read_text().split() if s)
    return set()


def save_cache_marker(symbols: set[str]):
    CACHE_SYMBOLS_MARKER.parent.mkdir(parents=True, exist_ok=True)
    CACHE_SYMBOLS_MARKER.write_text("\n".join(sorted(symbols)))


def update_cache(symbols: list[str], demo: bool, sleep_s: float) -> tuple[pd.DataFrame, int, int]:
    watch_set = set(symbols)
    cache = load_cache()
    have_dates_cached = set(cache["date"].dt.date) if not cache.empty else set()

    prior_watch_set = load_cache_marker()
    new_symbols = watch_set - prior_watch_set

    today = datetime.now(timezone.utc).date()

    if cache.empty:
        lookback_days = BACKFILL_CALENDAR_DAYS
        have_dates = set()
        log.info("No existing cache found - backfilling ~%d calendar days (first run only).", lookback_days)
    elif new_symbols:
        # The watchlist grew since the cache was last built. A date already
        # being in the cache doesn't mean it covers these new symbols - it
        # was fetched back when the watchlist was smaller. Force a full
        # re-backfill so new symbols get real history, not just 1-2 days.
        lookback_days = BACKFILL_CALENDAR_DAYS
        have_dates = set()
        log.info("Watchlist grew by %d symbol(s) since the cache was last built - forcing a "
                  "full ~%d-day re-backfill so they get proper history too (one-time cost; "
                  "symbols already covered are just re-fetched, which is harmless).",
                  len(new_symbols), lookback_days)
    else:
        have_dates = have_dates_cached
        most_recent = max(have_dates_cached)
        lookback_days = (today - most_recent).days + CATCHUP_BUFFER_DAYS
        log.info("Existing cache found through %s, no new symbols - catching up %d day(s).",
                  most_recent, lookback_days)

    candidate_dates = [datetime.combine(today, datetime.min.time()) - timedelta(days=i)
                        for i in range(1, lookback_days + 1)]

    new_frames, fetched, skipped_weekend = [], 0, 0
    for d in candidate_dates:
        if d.date() in have_dates:
            continue
        if d.weekday() >= 5:  # Sat/Sun - NSE never publishes, don't bother requesting
            skipped_weekend += 1
            continue

        raw = download_bhavcopy_demo(d, symbols, seed_offset=0) if demo else download_bhavcopy_live(d)
        if raw is None:
            continue  # holiday, or not yet published, or a transient miss - handled next run

        normalized = normalize_bhavcopy(raw, watch_set)
        if normalized is not None and not normalized.empty:
            new_frames.append(normalized)
            fetched += 1

        if not demo and sleep_s > 0:
            time.sleep(sleep_s + random.uniform(0, 0.3))

    if new_frames:
        cache = pd.concat([cache] + new_frames, ignore_index=True)
        cache = cache.drop_duplicates(subset=["symbol", "date"], keep="last")

    cutoff = pd.Timestamp(today - timedelta(days=CACHE_RETENTION_DAYS))
    cache = cache[cache["date"] >= cutoff]
    save_cache(cache)
    save_cache_marker(watch_set)

    log.info("Cache updated: %d new day(s) fetched, %d weekend day(s) skipped, %d total rows now cached.",
              fetched, skipped_weekend, len(cache))
    return cache, fetched, skipped_weekend


# ---------------------------------------------------------------------------
# Signal computation (unchanged from v1)
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame, z_threshold: float, price_move_threshold: float,
                    churn_vol_ratio: float, min_delivered_value_cr: float) -> dict | None:
    df = df.sort_values("date").reset_index(drop=True)
    d = df["delivery_pct"]
    if len(df) < MIN_ROWS_FOR_20D:
        return None

    mean20 = d.rolling(20).mean().shift(1)
    std20 = d.rolling(20).std().shift(1)
    z20 = (d - mean20) / std20.replace(0, np.nan)

    has_60d = len(df) >= MIN_ROWS_FOR_60D
    if has_60d:
        mean60 = d.rolling(60).mean().shift(1)
        std60 = d.rolling(60).std().shift(1)
        z60 = (d - mean60) / std60.replace(0, np.nan)
    else:
        z60 = pd.Series([np.nan] * len(df))

    vol = df["traded_qty"]
    vol_mean20 = vol.rolling(20).mean().shift(1)
    vol_ratio = vol / vol_mean20.replace(0, np.nan)

    z20_latest = z20.iloc[-1]
    if pd.isna(z20_latest):
        return None

    z60_latest = z60.iloc[-1] if has_60d else None
    vol_ratio_latest = vol_ratio.iloc[-1]

    close, prev_close = df["close"].iloc[-1], df["prev_close"].iloc[-1]
    price_change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0.0

    delivered_qty = df["deliverable_qty"].iloc[-1]
    avg_price = df["avg_price"].iloc[-1] if "avg_price" in df.columns else np.nan
    price_for_value = avg_price if pd.notna(avg_price) and avg_price > 0 else close
    delivered_value_cr = float(delivered_qty * price_for_value) / 1e7 if pd.notna(delivered_qty) else None
    low_liquidity = delivered_value_cr is not None and delivered_value_cr < min_delivered_value_cr

    if z20_latest >= z_threshold:
        if price_change_pct >= price_move_threshold:
            signal = "confirmed_accumulation"
        elif price_change_pct <= -price_move_threshold:
            signal = "possible_distribution"
        else:
            signal = "quiet_accumulation"
    elif vol_ratio_latest is not None and not pd.isna(vol_ratio_latest) and vol_ratio_latest >= churn_vol_ratio and z20_latest <= 0:
        signal = "speculative_churn"
    else:
        signal = "no_signal"

    sparkline = [round(float(x), 1) for x in d.tail(20).tolist()]

    return {
        "last_close": round(float(close), 2),
        "price_change_pct": round(float(price_change_pct), 2),
        "delivery_pct": round(float(d.iloc[-1]), 2),
        "delivery_avg_20d": round(float(mean20.iloc[-1]), 2) if not pd.isna(mean20.iloc[-1]) else None,
        "delivery_zscore_20d": round(float(z20_latest), 2),
        "delivery_zscore_60d": round(float(z60_latest), 2) if z60_latest is not None and not pd.isna(z60_latest) else None,
        "vol_ratio_20d": round(float(vol_ratio_latest), 2) if not pd.isna(vol_ratio_latest) else None,
        "delivered_value_cr": round(delivered_value_cr, 2) if delivered_value_cr is not None else None,
        "low_liquidity": bool(low_liquidity),
        "signal": signal,
        "delivery_sparkline_20d": sparkline,
        "sessions_available": len(df),
    }


def score_row(row: dict) -> float:
    score = 0.0
    z20 = row["delivery_zscore_20d"]
    if z20 > 0:
        score += z20 * 12
    z60 = row["delivery_zscore_60d"]
    if z60 is not None and z60 > 1.0:
        score += 8
    if row["vol_ratio_20d"] and row["vol_ratio_20d"] > 1.3 and row["signal"] != "speculative_churn":
        score += min((row["vol_ratio_20d"] - 1) * 6, 12)
    if row["signal"] == "speculative_churn":
        score = max(score, min(row["vol_ratio_20d"] or 0, 10) * 2)
    if row["low_liquidity"]:
        score *= 0.4
    # Corroboration bonus: a second, independently-sourced NSE dataset
    # (disclosed bulk/block deals) agreeing with the delivery-based read
    # is meaningfully stronger evidence than the delivery math alone.
    if row.get("smart_money_corroborated"):
        score += 12
    return round(max(score, 0), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_watchlist(path: Path) -> list[str]:
    if not path.exists():
        log.error("Watchlist not found at %s", path)
        sys.exit(1)
    return [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ---------------------------------------------------------------------------
# Cross-tool context: Smart Money Feed corroboration (sibling file read,
# zero extra network calls). A "quiet accumulation" read is more convincing
# if there was ALSO a disclosed bulk/block deal in that stock recently -
# two independently-sourced NSE datasets agreeing, not just one tool's math.
# ---------------------------------------------------------------------------

def load_smart_money_corroboration() -> dict[str, dict]:
    path = ROOT.parent / "smart-money" / "data" / "scan_results.json"
    if not path.exists():
        log.info("No smart-money output found at %s - corroboration flags will be blank this run.", path)
        return {}
    try:
        data = json.loads(path.read_text())
        return {
            r["symbol"]: {
                "net_value_cr": r.get("net_value_cr", 0),
                "known_investors_involved": r.get("known_investors_involved", []),
                "deal_count": r.get("deal_count", 0),
            }
            for r in data.get("symbol_summary", [])
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read smart-money output: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Signal history log + self-backtest. Uses bhavcopy_cache, which ALREADY
# holds historical close prices for every symbol - no new network calls
# needed to validate this tool's own past calls.
# ---------------------------------------------------------------------------

SIGNAL_LOG_PATH = ROOT / "data" / "signal_log.csv"
SIGNAL_LOG_COLUMNS = ["date", "symbol", "signal"]
BACKTEST_HORIZONS = (10, 20)
LOG_RETENTION_DAYS = 400


def load_signal_log() -> pd.DataFrame:
    if SIGNAL_LOG_PATH.exists():
        return pd.read_csv(SIGNAL_LOG_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)


def append_to_signal_log(rows: list[dict]) -> pd.DataFrame:
    log_df = load_signal_log()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    log_df = log_df[log_df["date"] != today]
    if rows:
        new_df = pd.DataFrame(rows)
        new_df["date"] = today
        log_df = pd.concat([log_df, new_df], ignore_index=True)
    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - pd.Timedelta(days=LOG_RETENTION_DAYS))
    log_df = log_df[log_df["date"] >= cutoff]
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.sort_values(["date", "symbol"]).to_csv(SIGNAL_LOG_PATH, index=False)
    return log_df


def backtest_signal_log(signal_log: pd.DataFrame, cache: pd.DataFrame) -> dict:
    if signal_log.empty or cache.empty:
        return {}

    by_signal: dict[str, dict[int, list[float]]] = {}
    for _, row in signal_log.iterrows():
        symbol, sig, log_date = row["symbol"], row["signal"], row["date"]
        if sig in ("no_signal",):
            continue
        prices = cache[cache["symbol"] == symbol].sort_values("date")
        prices_after = prices[prices["date"] >= log_date]
        if prices_after.empty:
            continue
        baseline_price = float(prices_after["close"].iloc[0])

        for horizon in BACKTEST_HORIZONS:
            future = prices_after.iloc[1:]
            if len(future) < horizon:
                continue
            fwd_price = float(future["close"].iloc[horizon - 1])
            ret = (fwd_price - baseline_price) / baseline_price * 100
            by_signal.setdefault(sig, {}).setdefault(horizon, []).append(ret)

    summary = {}
    for sig, horizons in by_signal.items():
        summary[sig] = {}
        for horizon, rets in horizons.items():
            wins = sum(1 for r in rets if r > 0)
            summary[sig][f"avg_return_{horizon}d"] = round(sum(rets) / len(rets), 2)
            summary[sig][f"hit_rate_{horizon}d"] = round(wins / len(rets) * 100, 1)
            summary[sig][f"n_{horizon}d"] = len(rets)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Delivery % / Volume Anomaly Radar (bhavcopy edition)")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--z-threshold", type=float, default=1.5)
    parser.add_argument("--price-move-threshold", type=float, default=2.0)
    parser.add_argument("--churn-vol-ratio", type=float, default=2.0)
    parser.add_argument("--min-delivered-value-cr", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    symbols = load_watchlist(args.watchlist)
    log.info("Loaded %d symbols from %s (demo=%s)", len(symbols), args.watchlist, args.demo)

    cache, fetched_days, _ = update_cache(symbols, demo=args.demo, sleep_s=args.sleep)
    corroboration = load_smart_money_corroboration()

    results, errors = [], []
    for symbol in symbols:
        sub = cache[cache["symbol"] == symbol]
        if sub.empty:
            errors.append({"symbol": symbol, "reason": "no cached bhavcopy rows for this symbol"})
            continue
        sig = compute_signal(sub, args.z_threshold, args.price_move_threshold,
                              args.churn_vol_ratio, args.min_delivered_value_cr)
        if sig is None:
            errors.append({"symbol": symbol, "reason": f"only {len(sub)} cached session(s), need {MIN_ROWS_FOR_20D}+"})
            continue

        row = {"symbol": symbol, **sig}
        smart_money = corroboration.get(symbol)
        row["smart_money_corroborated"] = bool(
            smart_money and row["signal"] in ("quiet_accumulation", "confirmed_accumulation")
            and (smart_money["net_value_cr"] > 0 or smart_money["known_investors_involved"])
        )
        row["smart_money_deal_count"] = smart_money["deal_count"] if smart_money else 0
        row["score"] = score_row(row)
        results.append(row)
        log.info("%-12s  signal=%-22s  z20=%5.2f  corroborated=%s  score=%.1f",
                  symbol, row["signal"], row["delivery_zscore_20d"], row["smart_money_corroborated"], row["score"])

    results.sort(key=lambda r: r["score"], reverse=True)

    todays_signal_rows = [
        {"symbol": r["symbol"], "signal": r["signal"]}
        for r in results if r["signal"] != "no_signal"
    ]
    signal_log = append_to_signal_log(todays_signal_rows)
    track_record = backtest_signal_log(signal_log, cache)
    log.info("Signal log: %d logged instance(s), track record covers %d signal type(s)",
              len(signal_log), len(track_record))

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(symbols),
        "success_count": len(results),
        "error_count": len(errors),
        "errors": errors,
        "cache_days_fetched_this_run": fetched_days,
        "cache_total_days": int(cache["date"].nunique()) if not cache.empty else 0,
        "track_record": track_record,
        "signal_log_size": len(signal_log),
        "params": {
            "z_threshold": args.z_threshold,
            "price_move_threshold_pct": args.price_move_threshold,
            "churn_vol_ratio_threshold": args.churn_vol_ratio,
            "min_delivered_value_cr": args.min_delivered_value_cr,
        },
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d results, %d errors)", args.out, output["success_count"], output["error_count"])


if __name__ == "__main__":
    main()
