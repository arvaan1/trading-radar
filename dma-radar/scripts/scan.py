#!/usr/bin/env python3
"""
DMA Crossover + Multi-Indicator Radar
--------------------------------------
Scans a watchlist of NSE symbols, computes a stack of technical indicators
(50/200 DMA gap + trend, RSI, MACD, ADX, Bollinger, Supertrend, volume,
and NSE delivery %), and writes a ranked JSON file the dashboard reads.

Usage:
    python scan.py                  # live run (NSE only, via the nse package)
    python scan.py --demo           # offline run with synthetic data,
                                     # useful for testing the pipeline
                                     # and the dashboard without network
                                     # access (e.g. sanity-checking before
                                     # wiring up GitHub Actions).
    python scan.py --watchlist path/to/file.txt

Data sources (all free, no API key required):
    - Price/volume history: NSE's own historical-data endpoint, via the
      nse package (nse.fetch_equity_historical_data). Previously Yahoo
      Finance - switched entirely to NSE after Yahoo caused two separate
      real bugs in this tool (a timezone crash, and very plausibly a
      stale-data incident too). Worth knowing: NSE's data here is raw,
      unadjusted for stock splits/bonuses - Yahoo's was adjusted. A split
      during the lookback window could in principle cause a false crossover
      signal around that date; there's no clean way to fully rule this out
      from NSE data alone.
    - Nifty 500 index history: same NSE endpoint family, via
      nse.fetch_historical_index_data - used for Relative Strength.
    - Delivery %: read directly from Delivery Radar's own output (a
      sibling-folder file read, not a separate fetch of the same data).

This script is designed to be run on a schedule (see
.github/workflows/update.yml) where it commits data/scan_results.json
back to the repo. The dashboard (index.html) is a static page that reads
that JSON file - no backend, no server cost.
"""

import argparse
import json
import sys
import time
import random
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("dma-radar")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
OUTPUT_PATH = ROOT / "data" / "scan_results.json"
NSE_CACHE_DIR = ROOT / ".nse_cache"  # session/cookie cache for the nse library, same pattern as every other tool

# ---------------------------------------------------------------------------
# Indicator math (implemented directly on pandas - no extra TA dependency)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + mult * std
    lower = sma - mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Returns +1 for uptrend, -1 for downtrend at each bar."""
    hl2 = (df["High"] + df["Low"]) / 2
    atr_s = atr(df, period)
    upperband = hl2 + mult * atr_s
    lowerband = hl2 - mult * atr_s
    direction = pd.Series(index=df.index, dtype="float64")
    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(len(df)):
        if i == 0:
            direction.iat[i] = 1
            continue
        close_prev = df["Close"].iat[i - 1]
        if upperband.iat[i] < final_upper.iat[i - 1] or close_prev > final_upper.iat[i - 1]:
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if lowerband.iat[i] > final_lower.iat[i - 1] or close_prev < final_lower.iat[i - 1]:
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        close_now = df["Close"].iat[i]
        if direction.iat[i - 1] == 1 and close_now < final_lower.iat[i]:
            direction.iat[i] = -1
        elif direction.iat[i - 1] == -1 and close_now > final_upper.iat[i]:
            direction.iat[i] = 1
        else:
            direction.iat[i] = direction.iat[i - 1]

    return direction


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_price_history_live(symbol: str, nse) -> pd.DataFrame | None:
    """Full daily OHLCV directly from NSE's own historical-data endpoint,
    via an already-open shared session (passed in, not opened per symbol -
    opening a fresh session per symbol for ~750 symbols would mean ~750
    redundant cookie handshakes; NSE's data now comes exclusively from
    NSE itself, replacing Yahoo Finance entirely, per explicit request
    after Yahoo caused two separate real bugs this session (a timezone
    crash, and very plausibly the stale-data incident too).

    HONEST CAVEAT, worth remembering: this is raw, UNADJUSTED price data.
    Yahoo Finance's auto_adjust=True gave split/bonus-adjusted prices;
    NSE's own historical endpoint does not adjust for corporate actions.
    If a stock in the universe splits or issues a bonus during the
    lookback window, its raw price series will show an overnight gap
    that isn't a real price move - which could, in principle, trigger a
    false DMA crossover around that date. There's no clean way to fully
    eliminate this risk from NSE data alone."""
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=460)  # ~15 months + buffer for weekends/holidays, same lookback as before

    try:
        raw = nse.fetch_equity_historical_data(symbol, from_date=from_date, to_date=to_date)
    except Exception as exc:  # noqa: BLE001
        log.debug("NSE history fetch failed for %s: %s", symbol, exc)
        return None

    if not raw or len(raw) < 210:
        return None

    rows = []
    for r in raw:
        try:
            rows.append({
                "date": pd.to_datetime(r["CH_TIMESTAMP"]),
                "Open": float(r["CH_OPENING_PRICE"]),
                "High": float(r["CH_TRADE_HIGH_PRICE"]),
                "Low": float(r["CH_TRADE_LOW_PRICE"]),
                "Close": float(r["CH_CLOSING_PRICE"]),
                "Volume": float(r["CH_TOT_TRADED_QTY"]),
            })
        except (KeyError, ValueError, TypeError):
            continue

    if len(rows) < 210:
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date").set_index("date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_delivery_pct_live(symbol: str) -> dict:
    """Delivery % now comes from Delivery Radar's own cache directly (a
    sibling file read, no network call) instead of a separate, fragile
    fetch of the same data. This used to call NSE's per-symbol API via
    nselib - the same unreliable pattern Delivery Radar itself abandoned
    for the bhavcopy approach - which is exactly why this column has been
    mostly blank. Reusing Delivery Radar's already-computed, already-
    validated numbers fixes that and removes a redundant data source
    entirely. See merge_delivery_context() for the actual lookup."""
    return {"delivery_pct": None, "delivery_zscore": None}


def fetch_price_history_demo(symbol: str, seed: int) -> pd.DataFrame:
    """Synthetic random-walk OHLCV for offline testing of the pipeline."""
    rng = np.random.default_rng(seed)
    n = 300
    # Requests a small buffer beyond n and slices to exactly n, rather than
    # trusting periods=n directly - pd.bdate_range(end=..., periods=n)
    # silently returns n-1 dates whenever the end anchor itself falls on a
    # weekend (confirmed: this returns 299 for periods=300 on a Saturday or
    # Sunday end date, 300 on a weekday). That's exactly why this never
    # surfaced in any earlier testing this week - every previous test run
    # happened to fall on a weekday. Slicing the tail sidesteps the anchor
    # question entirely instead of reasoning about pandas' exact behavior.
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n + 5)[-n:]
    drift = rng.normal(0.0003, 0.018, n)
    close = 100 * np.exp(np.cumsum(drift))
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(50_000, 2_000_000, n).astype(float)
    # inject a volume spike near the end for a couple of symbols
    if seed % 4 == 0:
        vol[-3:] *= rng.uniform(2.5, 4.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates
    )


def fetch_delivery_pct_demo(seed: int) -> dict:
    rng = np.random.default_rng(seed + 999)
    latest = float(rng.uniform(25, 70))
    z = float(rng.normal(0, 1))
    return {"delivery_pct": round(latest, 2), "delivery_zscore": round(z, 2)}


# ---------------------------------------------------------------------------
# Cross-tool context (sibling-folder file reads, zero extra network calls -
# this is the actual "interconnected web" mechanism: each tool's already-
# computed output becomes an input to the others, for free)
# ---------------------------------------------------------------------------

def load_delivery_context() -> dict[str, dict]:
    """Reads Delivery Radar's own latest output directly - replaces this
    tool's old broken standalone delivery fetch. Returns {} gracefully if
    Delivery Radar hasn't run yet or isn't present (e.g. during isolated
    testing), never raises."""
    path = ROOT.parent / "delivery-radar" / "data" / "scan_results.json"
    if not path.exists():
        log.info("No delivery-radar output found at %s - Delivery % will be blank this run.", path)
        return {}
    try:
        data = json.loads(path.read_text())
        return {
            r["symbol"]: {"delivery_pct": r.get("delivery_pct"), "delivery_zscore": r.get("delivery_zscore_20d")}
            for r in data.get("results", [])
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read delivery-radar output: %s", exc)
        return {}


def load_short_interest_context() -> dict[str, dict]:
    """Reads Smart Money Feed's short-selling data - lets a death-cross
    candidate show whether it's also being actively shorted, which is
    corroborating evidence a pure technical scanner has no way to know
    about on its own."""
    path = ROOT.parent / "smart-money" / "data" / "scan_results.json"
    if not path.exists():
        log.info("No smart-money output found at %s - short-interest context will be blank this run.", path)
        return {}
    try:
        data = json.loads(path.read_text())
        return {
            r["symbol"]: {"short_qty_total": r.get("short_qty_total", 0), "short_days": r.get("short_days", 0)}
            for r in data.get("symbol_summary", [])
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read smart-money output: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Nifty 500 index history, for Relative Strength
# ---------------------------------------------------------------------------

def fetch_index_history_live(nse) -> pd.Series | None:
    """Nifty 500 daily close, straight from NSE's own historical index
    endpoint - same reasoning and same shared-session pattern as
    fetch_price_history_live above."""
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=460)

    try:
        raw = nse.fetch_historical_index_data("NIFTY 500", from_date=from_date, to_date=to_date)
    except Exception as exc:  # noqa: BLE001
        log.warning("NSE Nifty 500 index fetch failed: %s", exc)
        return None

    if not raw or len(raw) < 70:
        return None

    rows = []
    for r in raw:
        try:
            d = pd.to_datetime(r["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
            if pd.notna(d):
                rows.append({"date": d, "close": float(r["EOD_CLOSE_INDEX_VAL"])})
        except (KeyError, ValueError, TypeError):
            continue

    if len(rows) < 70:
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date").set_index("date")
    return df["close"]


def fetch_index_history_demo(seed: int = 999) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = 300
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n + 5)[-n:]  # see fetch_price_history_demo for why the buffer+slice
    drift = rng.normal(0.0002, 0.009, n)  # a calmer walk than individual stocks - an index is diversified
    close = 100 * np.exp(np.cumsum(drift))
    return pd.Series(close, index=dates)


def compute_relative_strength(stock_close: pd.Series, index_close: pd.Series) -> dict:
    """Stock's own return over N days minus the index's return over the
    identical N days. Positive = the stock outperformed the broad market,
    not just 'went up' - and 'went up while the market went down' is a
    genuinely different, stronger statement than 'went up.'"""
    out = {"rs_20d": None, "rs_60d": None}
    if index_close is None or len(stock_close) < 61 or len(index_close) < 61:
        return out

    aligned_index = index_close.reindex(stock_close.index, method="ffill")
    for label, window in (("rs_20d", 20), ("rs_60d", 60)):
        if len(stock_close) <= window or pd.isna(aligned_index.iloc[-window - 1]):
            continue
        stock_ret = (stock_close.iloc[-1] / stock_close.iloc[-window - 1] - 1) * 100
        index_ret = (aligned_index.iloc[-1] / aligned_index.iloc[-window - 1] - 1) * 100
        out[label] = round(float(stock_ret - index_ret), 2)
    return out


# ---------------------------------------------------------------------------
# Signal history log + self-backtest (uses price data this run ALREADY
# fetched - no additional network calls needed to validate past calls)
# ---------------------------------------------------------------------------

SIGNAL_LOG_PATH = ROOT / "data" / "signal_log.csv"
SIGNAL_LOG_COLUMNS = ["date", "symbol", "signal", "gap_pct"]
BACKTEST_HORIZONS = (10, 20)  # trading days forward to check
LOG_RETENTION_DAYS = 400


def load_signal_log() -> pd.DataFrame:
    if SIGNAL_LOG_PATH.exists():
        return pd.read_csv(SIGNAL_LOG_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)


def append_to_signal_log(rows: list[dict]) -> pd.DataFrame:
    log_df = load_signal_log()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    log_df = log_df[log_df["date"] != today]  # today's entries get replaced, not duplicated, on a rerun
    if rows:
        new_df = pd.DataFrame(rows)
        new_df["date"] = today
        log_df = pd.concat([log_df, new_df], ignore_index=True)
    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - pd.Timedelta(days=LOG_RETENTION_DAYS))
    log_df = log_df[log_df["date"] >= cutoff]
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.sort_values(["date", "symbol"]).to_csv(SIGNAL_LOG_PATH, index=False)
    return log_df


def backtest_signal_log(signal_log: pd.DataFrame, price_histories: dict[str, pd.Series]) -> dict:
    """For each logged signal old enough to have forward data, computes
    the stock's return N trading days later using the price history this
    SAME run already fetched for that symbol (that history covers ~15
    months back, so it naturally covers past logged dates - zero extra
    fetches). Aggregates into a track record per signal type."""
    if signal_log.empty:
        return {}

    by_signal: dict[str, dict[int, list[float]]] = {}
    today = pd.Timestamp(datetime.now(timezone.utc).date())

    for _, row in signal_log.iterrows():
        symbol, sig, log_date = row["symbol"], row["signal"], row["date"]
        if sig == "no_signal" or symbol not in price_histories:
            continue
        prices = price_histories[symbol]
        prices_after = prices[prices.index >= log_date]
        if prices_after.empty:
            continue
        baseline_price = float(prices_after.iloc[0])

        for horizon in BACKTEST_HORIZONS:
            future = prices_after.iloc[1:]
            if len(future) < horizon:
                continue  # not enough time has passed yet for this specific logged signal
            fwd_price = float(future.iloc[horizon - 1])
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


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame, index_close: pd.Series | None) -> dict:
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    if pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1]):
        return None

    gap_pct_now = float((sma50.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1] * 100)
    gap_pct_5d = float((sma50.iloc[-6] - sma200.iloc[-6]) / sma200.iloc[-6] * 100) if len(sma50) > 210 else gap_pct_now
    narrowing = abs(gap_pct_now) < abs(gap_pct_5d)

    # detect a confirmed cross within the last 5 sessions
    recent_sign = np.sign((sma50 - sma200).dropna().iloc[-6:])
    confirmed_cross = recent_sign.nunique() > 1

    if confirmed_cross:
        signal = "confirmed_golden_cross" if gap_pct_now > 0 else "confirmed_death_cross"
    elif abs(gap_pct_now) < 2.0 and narrowing:
        signal = "approaching_golden_cross" if gap_pct_now < 0 else "approaching_death_cross"
    else:
        signal = "no_signal"

    rsi14 = float(rsi(close).iloc[-1])
    macd_line, macd_signal, macd_hist = macd(close)
    adx14 = float(adx(df).iloc[-1])
    _, _, bb_bw = bollinger(close)
    bb_bandwidth = float(bb_bw.iloc[-1]) if not pd.isna(bb_bw.iloc[-1]) else None
    st_dir = supertrend(df)
    supertrend_dir = "up" if st_dir.iloc[-1] == 1 else "down"

    vol = df["Volume"]
    vol_sma20 = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / vol_sma20.iloc[-1]) if vol_sma20.iloc[-1] else None

    rs = compute_relative_strength(close, index_close)

    return {
        "last_close": round(float(close.iloc[-1]), 2),
        "sma50": round(float(sma50.iloc[-1]), 2),
        "sma200": round(float(sma200.iloc[-1]), 2),
        "gap_pct": round(gap_pct_now, 3),
        "narrowing": bool(narrowing),
        "signal": signal,
        "rsi14": round(rsi14, 1),
        "macd_hist": round(float(macd_hist.iloc[-1]), 3),
        "adx14": round(adx14, 1),
        "bb_bandwidth": round(bb_bandwidth, 4) if bb_bandwidth is not None else None,
        "supertrend": supertrend_dir,
        "vol_ratio_20d": round(vol_ratio, 2) if vol_ratio is not None else None,
        "rs_20d": rs["rs_20d"],
        "rs_60d": rs["rs_60d"],
    }


def score_row(row: dict) -> float:
    """Higher = more actionable. Rewards proximity to a cross, narrowing
    momentum, trend-strength confirmation (ADX), volume confirmation,
    delivery-% confirmation, relative-strength confirmation, and - for
    bearish setups specifically - corroborating short-interest activity."""
    score = 0.0
    signal = row["signal"]

    if signal in ("confirmed_golden_cross", "confirmed_death_cross"):
        score += 40
    elif signal in ("approaching_golden_cross", "approaching_death_cross"):
        score += 25 + max(0, (2.0 - abs(row["gap_pct"])) * 5)  # closer gap = higher

    if row["narrowing"]:
        score += 8

    bullish = "golden" in signal
    if signal != "no_signal":
        if bullish and row["rsi14"] > 50:
            score += 6
        if (not bullish) and row["rsi14"] < 50:
            score += 6
        if bullish and row["supertrend"] == "up":
            score += 6
        if (not bullish) and row["supertrend"] == "down":
            score += 6

    if row["adx14"] is not None and row["adx14"] > 20:
        score += min(row["adx14"] / 5, 10)

    if row["vol_ratio_20d"] and row["vol_ratio_20d"] > 1.3:
        score += min((row["vol_ratio_20d"] - 1) * 10, 10)

    if row.get("delivery_zscore") is not None:
        if bullish and row["delivery_zscore"] > 0.5:
            score += min(row["delivery_zscore"] * 4, 10)

    # Relative strength: a golden cross on a stock already beating the
    # market is stronger; a death cross on a stock already lagging the
    # market is a stronger short candidate. Same bonus shape for both
    # directions, on purpose - this tool is built for someone who trades
    # both sides, not just longs.
    if signal != "no_signal" and row.get("rs_20d") is not None:
        if bullish and row["rs_20d"] > 0:
            score += min(row["rs_20d"] / 2, 10)
        if (not bullish) and row["rs_20d"] < 0:
            score += min(abs(row["rs_20d"]) / 2, 10)

    # Short-interest corroboration, bearish setups only: active short
    # selling alongside a death-cross-family signal is a second, entirely
    # independent data source (Smart Money Feed's disclosed short-sell
    # data) agreeing with the technical read.
    if (not bullish) and signal != "no_signal" and row.get("short_days"):
        score += min(row["short_days"] * 2, 10)

    return round(score, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_watchlist(path: Path) -> list[str]:
    if not path.exists():
        log.error("Watchlist not found at %s", path)
        sys.exit(1)
    symbols = []
    for line in path.read_text().splitlines():
        line = line.strip().upper()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)
    return symbols


def run(symbols: list[str], demo: bool, sleep_s: float, skip_delivery: bool) -> tuple[dict, dict[str, pd.Series]]:
    from contextlib import nullcontext

    results = []
    errors = []
    price_histories: dict[str, pd.Series] = {}

    delivery_context = load_delivery_context()
    short_context = load_short_interest_context()

    # One shared NSE session for the whole run, not one per symbol - with
    # ~750 symbols, opening a fresh session per symbol would mean ~750
    # redundant cookie handshakes. nullcontext keeps --demo mode simple
    # (no real session needed at all, nse=None, unused by the demo path).
    if demo:
        session_ctx = nullcontext(None)
    else:
        from nse import NSE
        NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        session_ctx = NSE(str(NSE_CACHE_DIR), server=True)

    with session_ctx as nse:
        index_close = fetch_index_history_demo() if demo else fetch_index_history_live(nse)
        if index_close is None:
            log.warning("No Nifty 500 index history available - Relative Strength will be blank this run.")

        for i, symbol in enumerate(symbols):
            try:
                if demo:
                    df = fetch_price_history_demo(symbol, seed=i)
                else:
                    df = fetch_price_history_live(symbol, nse)

                if df is None:
                    errors.append({"symbol": symbol, "reason": "insufficient price history"})
                    continue

                sig = compute_signal(df, index_close)
                if sig is None:
                    errors.append({"symbol": symbol, "reason": "could not compute 50/200 DMA"})
                    continue

                price_histories[symbol] = df["Close"]

                if skip_delivery:
                    delivery = {"delivery_pct": None, "delivery_zscore": None}
                else:
                    delivery = delivery_context.get(symbol, {"delivery_pct": None, "delivery_zscore": None})

                short_info = short_context.get(symbol, {"short_qty_total": 0, "short_days": 0})

                row = {"symbol": symbol, **sig, **delivery, **short_info}
                row["score"] = score_row(row)
                results.append(row)

                log.info("%-12s  signal=%-24s  gap=%6.2f%%  rs20=%s  score=%.1f",
                          symbol, row["signal"], row["gap_pct"], row.get("rs_20d"), row["score"])

            except Exception as exc:  # noqa: BLE001
                log.warning("Failed on %s: %s", symbol, exc)
                errors.append({"symbol": symbol, "reason": str(exc)})

            if not demo and sleep_s > 0 and i < len(symbols) - 1:
                time.sleep(sleep_s + random.uniform(0, 0.3))  # polite jitter, avoid hammering endpoints

    results.sort(key=lambda r: r["score"], reverse=True)

    # Surfaces the actual trading day the price data reflects, separate
    # from when the script happened to run. These are NOT the same fact,
    # and only ever showing "last scan time" was exactly what let a stale
    # fetch look indistinguishable from a fresh one. Yahoo Finance doesn't
    # always have a trading day's official close finalized within an hour
    # or two of market close - if it hadn't yet, this run would silently
    # capture the prior day's bar with no visible sign anything was off.
    latest_bar_dates = [str(s.index[-1].date()) for s in price_histories.values() if len(s) > 0]
    if latest_bar_dates:
        data_as_of = max(set(latest_bar_dates), key=latest_bar_dates.count)  # mode - the date most symbols agree on
        agreement_pct = round(latest_bar_dates.count(data_as_of) / len(latest_bar_dates) * 100, 1)
    else:
        data_as_of, agreement_pct = None, None

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_as_of": data_as_of,
        "data_as_of_agreement_pct": agreement_pct,
        "universe_size": len(symbols),
        "success_count": len(results),
        "error_count": len(errors),
        "errors": errors,
        "results": results,
    }
    return output, price_histories


def main():
    parser = argparse.ArgumentParser(description="DMA Crossover + Multi-Indicator Radar scanner")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--demo", action="store_true", help="Use synthetic data, no network calls")
    parser.add_argument("--sleep", type=float, default=0.6, help="Seconds between live requests")
    parser.add_argument("--skip-delivery", action="store_true", help="Skip the NSE delivery-%% lookup (faster)")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    symbols = load_watchlist(args.watchlist)
    log.info("Loaded %d symbols from %s (demo=%s)", len(symbols), args.watchlist, args.demo)

    output, price_histories = run(symbols, demo=args.demo, sleep_s=args.sleep, skip_delivery=args.skip_delivery)

    if output.get("data_as_of"):
        data_date = datetime.strptime(output["data_as_of"], "%Y-%m-%d").date()
        run_date = datetime.now(timezone.utc).date()
        days_behind = (run_date - data_date).days
        # More than 1 calendar day behind is worth a loud warning even
        # accounting for weekends, since this run log is the fastest place
        # to catch a stale-data-source problem before it reaches the
        # dashboard silently.
        if days_behind > 3:
            log.warning("DATA FRESHNESS: latest price data is from %s (%d days before this run) - "
                        "Yahoo Finance may not have published the latest close yet, or something else is stale.",
                        output["data_as_of"], days_behind)
        else:
            log.info("Data as of: %s (%s%% of symbols agree)", output["data_as_of"], output["data_as_of_agreement_pct"])

    # Log today's signals, then backtest the log using price data this run
    # already has in memory - no extra network calls to validate history.
    # Wrapped defensively on purpose: the backtest is a bonus layered on
    # top of the core scan, not the reason this tool exists. If it fails
    # for any reason - this bug or a future one - the actual scan results
    # (which just took several minutes to gather) still get written,
    # instead of the whole run failing and producing nothing at all.
    try:
        todays_signal_rows = [
            {"symbol": r["symbol"], "signal": r["signal"], "gap_pct": r["gap_pct"]}
            for r in output["results"] if r["signal"] != "no_signal"
        ]
        signal_log = append_to_signal_log(todays_signal_rows)
        track_record = backtest_signal_log(signal_log, price_histories)
        output["track_record"] = track_record
        output["signal_log_size"] = len(signal_log)
        log.info("Signal log: %d logged instance(s) total, track record covers %d signal type(s)",
                  len(signal_log), len(track_record))
    except Exception as exc:  # noqa: BLE001
        log.error("Signal log / backtest step failed (core scan results are unaffected): %s", exc)
        output["track_record"] = {}
        output["signal_log_size"] = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d results, %d errors)", args.out, output["success_count"], output["error_count"])


if __name__ == "__main__":
    main()
