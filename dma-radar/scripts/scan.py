#!/usr/bin/env python3
"""
DMA Crossover + Multi-Indicator Radar
--------------------------------------
Scans a watchlist of NSE symbols, computes a stack of technical indicators
(50/200 DMA gap + trend, RSI, MACD, ADX, Bollinger, Supertrend, volume,
and NSE delivery %), and writes a ranked JSON file the dashboard reads.

Usage:
    python scan.py                  # live run (yfinance + nselib)
    python scan.py --demo           # offline run with synthetic data,
                                     # useful for testing the pipeline
                                     # and the dashboard without network
                                     # access (e.g. sanity-checking before
                                     # wiring up GitHub Actions).
    python scan.py --watchlist path/to/file.txt

Data sources (all free, no API key required):
    - Price/volume history: Yahoo Finance via yfinance (SYMBOL.NS)
    - Delivery %: NSE's public data via the nselib package

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("dma-radar")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
OUTPUT_PATH = ROOT / "data" / "scan_results.json"

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

def fetch_price_history_live(symbol: str) -> pd.DataFrame | None:
    import yfinance as yf

    ticker = f"{symbol}.NS"
    df = yf.Ticker(ticker).history(period="15mo", interval="1d", auto_adjust=True)
    if df is None or df.empty or len(df) < 210:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def fetch_delivery_pct_live(symbol: str) -> dict:
    """Returns latest delivery % and a 20-day z-score. Best-effort: NSE's
    endpoints are undocumented and occasionally rate-limit, so failures here
    should never crash the whole scan."""
    try:
        from nselib import capital_market

        df = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period="2M")
        if df is None or len(df) == 0:
            return {"delivery_pct": None, "delivery_zscore": None}

        # Column names from NSE's raw feed are inconsistent about spacing/case
        # across nselib versions, so match loosely.
        def find_col(candidates):
            norm = {c: "".join(c.lower().split()).replace("%", "").replace(".", "") for c in df.columns}
            for cand in candidates:
                cand_n = "".join(cand.lower().split()).replace("%", "").replace(".", "")
                for orig, n in norm.items():
                    if cand_n in n:
                        return orig
            return None

        col = find_col(["DlyQttoTradedQty", "DlyQtytoTradedQty", "PercentDeliverable"])
        if col is None:
            return {"delivery_pct": None, "delivery_zscore": None}

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 5:
            return {"delivery_pct": None, "delivery_zscore": None}

        latest = series.iloc[-1]
        mean, std = series.mean(), series.std()
        z = (latest - mean) / std if std and std > 0 else 0.0
        return {"delivery_pct": round(float(latest), 2), "delivery_zscore": round(float(z), 2)}
    except Exception as exc:  # noqa: BLE001
        log.debug("delivery fetch failed for %s: %s", symbol, exc)
        return {"delivery_pct": None, "delivery_zscore": None}


def fetch_price_history_demo(symbol: str, seed: int) -> pd.DataFrame:
    """Synthetic random-walk OHLCV for offline testing of the pipeline."""
    rng = np.random.default_rng(seed)
    n = 300
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
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
# Signal computation
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> dict:
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
    }


def score_row(row: dict) -> float:
    """Higher = more actionable. Rewards proximity to a cross, narrowing
    momentum, trend-strength confirmation (ADX), volume confirmation, and
    delivery-% confirmation when available."""
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


def run(symbols: list[str], demo: bool, sleep_s: float, skip_delivery: bool) -> dict:
    results = []
    errors = []

    for i, symbol in enumerate(symbols):
        try:
            if demo:
                df = fetch_price_history_demo(symbol, seed=i)
            else:
                df = fetch_price_history_live(symbol)

            if df is None:
                errors.append({"symbol": symbol, "reason": "insufficient price history"})
                continue

            sig = compute_signal(df)
            if sig is None:
                errors.append({"symbol": symbol, "reason": "could not compute 50/200 DMA"})
                continue

            if skip_delivery:
                delivery = {"delivery_pct": None, "delivery_zscore": None}
            elif demo:
                delivery = fetch_delivery_pct_demo(seed=i)
            else:
                delivery = fetch_delivery_pct_live(symbol)

            row = {"symbol": symbol, **sig, **delivery}
            row["score"] = score_row(row)
            results.append(row)

            log.info("%-12s  signal=%-24s  gap=%6.2f%%  score=%.1f", symbol, row["signal"], row["gap_pct"], row["score"])

        except Exception as exc:  # noqa: BLE001
            log.warning("Failed on %s: %s", symbol, exc)
            errors.append({"symbol": symbol, "reason": str(exc)})

        if not demo and sleep_s > 0 and i < len(symbols) - 1:
            time.sleep(sleep_s + random.uniform(0, 0.3))  # polite jitter, avoid hammering endpoints

    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(symbols),
        "success_count": len(results),
        "error_count": len(errors),
        "errors": errors,
        "results": results,
    }


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

    output = run(symbols, demo=args.demo, sleep_s=args.sleep, skip_delivery=args.skip_delivery)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d results, %d errors)", args.out, output["success_count"], output["error_count"])


if __name__ == "__main__":
    main()
