#!/usr/bin/env python3
"""
Delivery % / Volume Anomaly Radar
-----------------------------------
Flags NSE stocks where today's delivered-quantity percentage is a
statistical outlier vs. its own trailing 20/60-day average - the
"someone is quietly accumulating (or exiting) real shares, not just
churning volume intraday" detector.

Why delivery % and not just volume: a volume spike alone is ambiguous -
could be genuine interest, could be algos/operators trading with each
other and squaring off same-day. Delivery % tells you how much of that
volume actually settled into demat accounts. A stock's own history is the
baseline (not a market-wide number), because "normal" delivery % varies
hugely by stock - a sleepy small-cap might run 60% delivery on a typical
day while a high-beta trading favourite runs 15%. What matters is a stock
trading well outside *its own* normal range.

Usage:
    python scan.py                  # live run (nselib only - one data source)
    python scan.py --demo           # offline run, synthetic data, seeded to
                                     # exercise all four signal categories
    python scan.py --z-threshold 2.0 --min-delivered-value-cr 1.0

Data source: NSE's public price/volume/deliverable-position data via the
`nselib` package. This tool needs only that one endpoint - no yfinance,
no second source - which makes it lighter and less fragile than the DMA
radar's pipeline.
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
log = logging.getLogger("delivery-radar")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT / "watchlist.txt"
OUTPUT_PATH = ROOT / "data" / "scan_results.json"

MIN_ROWS_FOR_20D = 25   # need at least ~25 sessions to trust a 20d baseline
MIN_ROWS_FOR_60D = 65   # ~65 sessions to trust a 60d baseline

# ---------------------------------------------------------------------------
# Column normalization - nselib's column naming has varied slightly across
# versions ("% Dly Qt toTraded Qty" vs "%DlyQttoTradedQty" etc.), so match
# loosely on a normalized (lowercased, whitespace/punctuation-stripped) form
# rather than hardcoding one exact string.
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {c: _normalize(c) for c in df.columns}
    for cand in candidates:
        cand_n = _normalize(cand)
        for orig, n in norm_map.items():
            if cand_n in n:
                return orig
    return None


COLUMN_CANDIDATES = {
    "date": ["date"],
    "prev_close": ["prevclose"],
    "close": ["closeprice"],
    "close_fallback": ["lastprice"],
    "vwap": ["vwap"],
    "traded_qty": ["totaltradedquantity", "tradedquantity"],
    "deliverable_qty": ["deliverableqty", "deliverablequantity"],
    "delivery_pct": ["dlyqttotradedqty", "dlyqtytotradedqty", "percentdeliverable"],
}


def normalize_frame(raw: pd.DataFrame) -> pd.DataFrame | None:
    cols = {}
    for key in ("date", "prev_close", "vwap", "traded_qty", "deliverable_qty", "delivery_pct"):
        cols[key] = find_col(raw, COLUMN_CANDIDATES[key])
    cols["close"] = find_col(raw, COLUMN_CANDIDATES["close"]) or find_col(raw, COLUMN_CANDIDATES["close_fallback"])

    required = ("date", "prev_close", "close", "traded_qty", "deliverable_qty", "delivery_pct")
    if any(cols[k] is None for k in required):
        return None

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[cols["date"]], errors="coerce", dayfirst=True),
        "prev_close": pd.to_numeric(raw[cols["prev_close"]], errors="coerce"),
        "close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
        "vwap": pd.to_numeric(raw[cols["vwap"]], errors="coerce") if cols.get("vwap") else np.nan,
        "traded_qty": pd.to_numeric(raw[cols["traded_qty"]], errors="coerce"),
        "deliverable_qty": pd.to_numeric(raw[cols["deliverable_qty"]], errors="coerce"),
        "delivery_pct": pd.to_numeric(raw[cols["delivery_pct"]], errors="coerce"),
    })
    df = df.dropna(subset=["date", "close", "traded_qty", "delivery_pct"]).sort_values("date")
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_delivery_history_live(symbol: str) -> pd.DataFrame | None:
    from nselib import capital_market

    raw = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period="6M")
    if raw is None or len(raw) == 0:
        return None
    return normalize_frame(pd.DataFrame(raw))


def fetch_delivery_history_demo(symbol: str, seed: int) -> pd.DataFrame:
    """Synthetic delivery-% + price history. Seeded to rotate through all
    four signal categories so the dashboard can be visually checked against
    every badge type without needing a live NSE connection."""
    rng = np.random.default_rng(seed)
    n = 130
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    base_delivery = rng.uniform(30, 55)
    delivery_pct = np.clip(rng.normal(base_delivery, 6, n), 5, 95)

    price = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    traded_qty = rng.integers(80_000, 900_000, n).astype(float)

    scenario = seed % 5
    if scenario == 0:
        # quiet accumulation: delivery spike, price roughly flat
        delivery_pct[-1] = base_delivery + rng.uniform(22, 32)
        price[-1] = price[-2] * (1 + rng.uniform(-0.004, 0.006))
    elif scenario == 1:
        # confirmed accumulation: delivery spike + price up
        delivery_pct[-1] = base_delivery + rng.uniform(22, 32)
        price[-1] = price[-2] * (1 + rng.uniform(0.025, 0.045))
        traded_qty[-1] *= rng.uniform(1.5, 2.2)
    elif scenario == 2:
        # possible distribution: delivery spike + price down
        delivery_pct[-1] = base_delivery + rng.uniform(20, 30)
        price[-1] = price[-2] * (1 - rng.uniform(0.025, 0.045))
    elif scenario == 3:
        # speculative churn: volume spike, delivery % suppressed
        traded_qty[-1] *= rng.uniform(3.0, 5.0)
        delivery_pct[-1] = max(4, base_delivery - rng.uniform(18, 26))
    # scenario 4: left as normal/no-signal

    close = price
    prev_close = np.roll(price, 1)
    prev_close[0] = price[0] * 0.995

    df = pd.DataFrame({
        "date": dates,
        "prev_close": prev_close,
        "close": close,
        "vwap": close * (1 + rng.normal(0, 0.002, n)),
        "traded_qty": traded_qty,
        "deliverable_qty": traded_qty * (delivery_pct / 100),
        "delivery_pct": delivery_pct,
    })
    return df


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame, z_threshold: float, price_move_threshold: float,
                    churn_vol_ratio: float, min_delivered_value_cr: float) -> dict | None:
    d = df["delivery_pct"]
    if len(df) < MIN_ROWS_FOR_20D:
        return None

    # baseline uses PRIOR days only (shift(1)) so today's value can't inflate
    # its own baseline - this is what makes it a genuine outlier test.
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
        return None  # not enough usable history to trust a baseline

    z60_latest = z60.iloc[-1] if has_60d else None
    vol_ratio_latest = vol_ratio.iloc[-1]

    close, prev_close = df["close"].iloc[-1], df["prev_close"].iloc[-1]
    price_change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0.0

    delivered_qty = df["deliverable_qty"].iloc[-1]
    vwap = df["vwap"].iloc[-1]
    price_for_value = vwap if pd.notna(vwap) and vwap > 0 else close
    delivered_value_cr = float(delivered_qty * price_for_value) / 1e7 if pd.notna(delivered_qty) else None
    low_liquidity = delivered_value_cr is not None and delivered_value_cr < min_delivered_value_cr

    # --- classify ---
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
    }


def score_row(row: dict) -> float:
    """Higher = statistically stronger, more actionable anomaly. This scores
    the STRENGTH of the anomaly, not a buy/sell opinion - direction is left
    to the `signal` label so the ranking stays honest about what it does
    and doesn't know (delivery % alone can't distinguish real buyers from
    real sellers; price context is a hint, not proof)."""
    score = 0.0
    z20 = row["delivery_zscore_20d"]

    if z20 > 0:
        score += z20 * 12

    z60 = row["delivery_zscore_60d"]
    if z60 is not None and z60 > 1.0:
        score += 8  # two independent windows agreeing is stronger evidence than one

    if row["vol_ratio_20d"] and row["vol_ratio_20d"] > 1.3 and row["signal"] != "speculative_churn":
        score += min((row["vol_ratio_20d"] - 1) * 6, 12)

    if row["signal"] == "speculative_churn":
        score = max(score, min(row["vol_ratio_20d"] or 0, 10) * 2)  # rank churn on its own scale, low ceiling

    if row["low_liquidity"]:
        score *= 0.4  # don't let a thin, barely-traded stock dominate the ranking

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


def run(symbols: list[str], demo: bool, sleep_s: float, z_threshold: float,
        price_move_threshold: float, churn_vol_ratio: float, min_delivered_value_cr: float) -> dict:
    results, errors = [], []

    for i, symbol in enumerate(symbols):
        try:
            df = fetch_delivery_history_demo(symbol, seed=i) if demo else fetch_delivery_history_live(symbol)

            if df is None or len(df) == 0:
                errors.append({"symbol": symbol, "reason": "no usable delivery data"})
                continue

            sig = compute_signal(df, z_threshold, price_move_threshold, churn_vol_ratio, min_delivered_value_cr)
            if sig is None:
                errors.append({"symbol": symbol, "reason": f"fewer than {MIN_ROWS_FOR_20D} usable sessions"})
                continue

            row = {"symbol": symbol, **sig}
            row["score"] = score_row(row)
            results.append(row)

            log.info("%-12s  signal=%-22s  z20=%5.2f  vol=%4sx  score=%.1f",
                      symbol, row["signal"], row["delivery_zscore_20d"],
                      f'{row["vol_ratio_20d"]:.1f}' if row["vol_ratio_20d"] else "  - ", row["score"])

        except Exception as exc:  # noqa: BLE001
            log.warning("Failed on %s: %s", symbol, exc)
            errors.append({"symbol": symbol, "reason": str(exc)})

        if not demo and sleep_s > 0 and i < len(symbols) - 1:
            time.sleep(sleep_s + random.uniform(0, 0.3))

    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(symbols),
        "success_count": len(results),
        "error_count": len(errors),
        "errors": errors,
        "params": {
            "z_threshold": z_threshold,
            "price_move_threshold_pct": price_move_threshold,
            "churn_vol_ratio_threshold": churn_vol_ratio,
            "min_delivered_value_cr": min_delivered_value_cr,
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Delivery % / Volume Anomaly Radar")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--demo", action="store_true", help="Use synthetic data, no network calls")
    parser.add_argument("--sleep", type=float, default=0.6)
    parser.add_argument("--z-threshold", type=float, default=1.5, help="Delivery-%% z-score to count as an outlier")
    parser.add_argument("--price-move-threshold", type=float, default=2.0, help="Price move %% to call direction (else 'quiet')")
    parser.add_argument("--churn-vol-ratio", type=float, default=2.0, help="Vol-vs-20d ratio to flag speculative churn")
    parser.add_argument("--min-delivered-value-cr", type=float, default=0.5, help="Below this (Rs. crore), dampen the score as low-liquidity noise")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    symbols = load_watchlist(args.watchlist)
    log.info("Loaded %d symbols from %s (demo=%s)", len(symbols), args.watchlist, args.demo)

    output = run(symbols, demo=args.demo, sleep_s=args.sleep, z_threshold=args.z_threshold,
                 price_move_threshold=args.price_move_threshold, churn_vol_ratio=args.churn_vol_ratio,
                 min_delivered_value_cr=args.min_delivered_value_cr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d results, %d errors)", args.out, output["success_count"], output["error_count"])


if __name__ == "__main__":
    main()
