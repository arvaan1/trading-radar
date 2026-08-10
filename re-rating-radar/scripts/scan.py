#!/usr/bin/env python3
"""
Re-Rating Radar - a pre-re-rating detector
--------------------------------------------
WHAT THIS IS, HONESTLY: a systematic operationalization of a pattern
found in real, documented case-study research (Paras Defence's public
trading and financial history specifically) - NOT a statistically
backtested, weight-optimized model. That distinction matters enough to
repeat on every page of this tool's dashboard, not just here.

The pattern this tool looks for, grounded in real evidence: Paras
Defence spent a multi-year stretch (FY21-FY25) compounding revenue
(Rs 143 Cr -> Rs 365 Cr) and profit while debt went to zero - a real
fundamental trajectory the market had not yet priced in. Technically,
this showed up as an extended LOW-VOLUME price consolidation ("coiling"
- documented as a ~110-day base by outside technical analysts), which
then broke on volume roughly 10x its 20-day average. The order-win
headlines that followed were CONFIRMING/LAGGING signals - useful for
conviction, useless for early entry, since you can't front-run a filing
that hasn't happened yet.

What this tool actually screens for, using ONLY data this system
already has reliably (no new fragile NSE endpoint, no fundamental data
this system can't access):
    1. COMPRESSION: has this stock's own volatility (Bollinger
       bandwidth, already computed by DMA Radar) been unusually tight
       for its own recent history? Tracked via a small persistent cache
       this tool builds up itself, run over run - like every other
       signal-history cache in this system, this starts thin and gets
       more meaningful over the following weeks.
    2. QUIET OUTPERFORMANCE: is Relative Strength (DMA Radar) already
       positive, but not yet extreme? The sweet spot is "starting to
       lead, not yet spectacular" - RS already very high suggests the
       move may be well underway, not early.
    3. REAL ACCUMULATION: is Delivery Radar showing a genuine delivery-%
       outlier - shares actually settling into accounts, not just
       trading hands intraday?
    4. INFORMED PARTICIPATION: does Smart Money Feed show a tracked
       known investor or a repeat buyer in this name?
    5. THE TRANSITION ITSELF: was this stock compressed, and has volume
       just started expanding? That's the state-3-to-4 transition this
       tool is built to catch, not the aftermath.

Read the README before trusting this tool's numbers the way you'd trust
DMA Radar's or Delivery Radar's - this one is younger, built on a
single deeply-researched case study rather than a broad backtest, and
says so on every page.
"""

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("re-rating-radar")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = ROOT.parent / "dma-radar" / "watchlist.txt"  # reads DMA Radar's own watchlist directly - guarantees the exact same universe, zero risk of the two ever drifting apart, no separate copy to maintain
OUTPUT_PATH = ROOT / "data" / "scan_results.json"

COMPRESSION_LOG_PATH = ROOT / "data" / "compression_history.csv"
COMPRESSION_LOG_COLUMNS = ["date", "symbol", "bb_bandwidth"]
COMPRESSION_RETENTION_DAYS = 250
COMPRESSION_LOOKBACK_FOR_PERCENTILE = 60  # trading days of a symbol's own history needed for a meaningful percentile


# ---------------------------------------------------------------------------
# Cross-tool reads (sibling-folder file reads, zero extra network calls -
# this whole tool is built entirely out of data the other four already
# computed, plus one small thing of its own: compression duration)
# ---------------------------------------------------------------------------

def load_dma_context() -> dict[str, dict]:
    path = ROOT.parent / "dma-radar" / "data" / "scan_results.json"
    if not path.exists():
        log.warning("No dma-radar output found - this tool cannot run meaningfully without it.")
        return {}
    try:
        data = json.loads(path.read_text())
        return {r["symbol"]: r for r in data.get("results", [])}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read dma-radar output: %s", exc)
        return {}


def load_delivery_context() -> dict[str, dict]:
    path = ROOT.parent / "delivery-radar" / "data" / "scan_results.json"
    if not path.exists():
        log.info("No delivery-radar output found - accumulation context will be blank this run.")
        return {}
    try:
        data = json.loads(path.read_text())
        return {r["symbol"]: r for r in data.get("results", [])}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read delivery-radar output: %s", exc)
        return {}


def load_smart_money_context() -> dict[str, dict]:
    path = ROOT.parent / "smart-money" / "data" / "scan_results.json"
    if not path.exists():
        log.info("No smart-money output found - participation context will be blank this run.")
        return {}
    try:
        data = json.loads(path.read_text())
        return {r["symbol"]: r for r in data.get("symbol_summary", [])}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read smart-money output: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Compression history - the one thing this tool computes itself, built
# from DMA Radar's already-computed bb_bandwidth rather than a fresh
# price fetch, to avoid duplicating that tool's expensive ~750-symbol
# NSE call. Builds up meaningfully over the coming weeks, same honest
# "grows over time" pattern as every other cache in this system.
# ---------------------------------------------------------------------------

def load_compression_log() -> pd.DataFrame:
    if COMPRESSION_LOG_PATH.exists():
        return pd.read_csv(COMPRESSION_LOG_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=COMPRESSION_LOG_COLUMNS)


def update_compression_log(dma_context: dict[str, dict]) -> pd.DataFrame:
    log_df = load_compression_log()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    log_df = log_df[log_df["date"] != today]  # replace today's entries on a rerun, don't duplicate

    rows = []
    for symbol, r in dma_context.items():
        bw = r.get("bb_bandwidth")
        if bw is not None:
            rows.append({"date": today, "symbol": symbol, "bb_bandwidth": bw})
    if rows:
        log_df = pd.concat([log_df, pd.DataFrame(rows)], ignore_index=True)

    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=COMPRESSION_RETENTION_DAYS))
    log_df = log_df[log_df["date"] >= cutoff]
    COMPRESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.sort_values(["symbol", "date"]).to_csv(COMPRESSION_LOG_PATH, index=False)
    return log_df


def compute_compression_state(symbol: str, today_bandwidth: float | None, compression_log: pd.DataFrame) -> dict:
    """Where does TODAY's bandwidth sit relative to this symbol's own
    RECENT history? Low percentile = currently tight/coiling relative to
    its own normal range - not an absolute threshold, since a naturally
    low-volatility stock and a naturally choppy one have very different
    baselines.

    Deliberately limited to the last COMPRESSION_LOOKBACK_FOR_PERCENTILE
    rows, not everything the cache has stored - the cache itself keeps up
    to 250 calendar days for the retention window, but comparing today
    against volatility from 8-9 months ago answers a different, less
    useful question than "is this tight relative to its recent character."
    (This constant existed but was never actually applied here until this
    fix - worth being upfront about, not quietly patching.)"""
    if today_bandwidth is None:
        return {"percentile": None, "days_of_history": 0, "is_compressed": False}

    hist = compression_log[compression_log["symbol"] == symbol]["bb_bandwidth"].dropna()
    hist = hist.tail(COMPRESSION_LOOKBACK_FOR_PERCENTILE)  # most recent ~60 trading days only, not the full cache
    if len(hist) < 10:
        return {"percentile": None, "days_of_history": len(hist), "is_compressed": False}

    percentile = round(float((hist < today_bandwidth).mean() * 100), 1)
    return {
        "percentile": percentile,
        "days_of_history": len(hist),
        "is_compressed": percentile <= 25,  # today's bandwidth is in the tightest quarter of its own recent range
    }


# ---------------------------------------------------------------------------
# Scoring - transparent and reasoned, not statistically fitted. Every
# component here is explained in the README with the case-study logic
# behind it. This is deliberately NOT dressed up as more rigorous than
# it is.
# ---------------------------------------------------------------------------

def score_row(row: dict) -> float:
    score = 0.0

    # Compression: a stock currently in the tightest quarter of its own
    # recent volatility range is "coiling" - the base-building phase
    # that preceded Paras Defence's breakout.
    if row.get("compression_percentile") is not None:
        if row["compression_percentile"] <= 25:
            score += 25 * (1 - row["compression_percentile"] / 25)  # tighter = more points, up to 25

    # Quiet outperformance: positive but not yet extreme RS is the sweet
    # spot - already leading, not yet the story everyone knows.
    rs20 = row.get("rs_20d")
    if rs20 is not None:
        if 2 <= rs20 <= 20:
            score += 20
        elif 20 < rs20 <= 40:
            score += 10  # still worth something, but less "early"
        # RS above 40%: no points - this is likely already well past the early stage

    # Real accumulation: Delivery Radar's own z-score, reused directly.
    z20 = row.get("delivery_zscore_20d")
    if z20 is not None and z20 > 1.0:
        score += min(z20 * 5, 20)

    # Informed participation: known investor or repeat buyer presence.
    if row.get("known_investors_involved"):
        score += 15
    if row.get("repeat_buyers"):
        score += 10

    # The transition itself: was compressed AND volume is now expanding -
    # this is the state-3-to-4 shift this tool exists to catch.
    if row.get("is_compressed") and row.get("vol_ratio_20d") and row["vol_ratio_20d"] > 1.3:
        score += 15

    # Trend confirmation, small weight on purpose - this tool is about
    # catching the setup before it's obvious, not re-scoring what DMA
    # Radar's own crossover score already covers well.
    if row.get("adx14") is not None and row["adx14"] > 20:
        score += 5

    return round(min(score, 100), 1)


def classify_stage(row: dict) -> str:
    """A rough state label, in the spirit of the case-study framework -
    deliberately simple, not a rigorously validated state-transition
    model."""
    dma_signal = row.get("dma_signal") or ""
    rs20 = row.get("rs_20d")

    if "golden" in dma_signal and rs20 is not None and rs20 > 30:
        return "likely already broadly recognized - later stage, less early"
    if row.get("is_compressed") and (row.get("vol_ratio_20d") or 0) > 1.3:
        return "compression breaking now - the transition this tool targets"
    if row.get("is_compressed"):
        return "quiet compression - still coiling"
    if rs20 is not None and rs20 > 0:
        return "early outperformance, not yet compressed"
    return "no clear stage signal"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_watchlist(path: Path) -> list[str]:
    if not path.exists():
        log.error("Watchlist not found at %s", path)
        return []
    return [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser(description="Re-Rating Radar: pre-re-rating detector built from documented case-study research")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    symbols = load_watchlist(args.watchlist)
    log.info("Loaded %d symbols from %s (demo=%s)", len(symbols), args.watchlist, args.demo)

    if args.demo:
        dma_context = demo_dma_context(symbols)
        delivery_context = demo_delivery_context(symbols)
        smart_money_context = demo_smart_money_context(symbols)
    else:
        dma_context = load_dma_context()
        delivery_context = load_delivery_context()
        smart_money_context = load_smart_money_context()

    compression_log = update_compression_log(dma_context)

    results = []
    for symbol in symbols:
        dma = dma_context.get(symbol)
        if dma is None:
            continue  # this tool is fundamentally built on DMA Radar's output - nothing to say without it

        delivery = delivery_context.get(symbol, {})
        smart_money = smart_money_context.get(symbol, {})
        compression = compute_compression_state(symbol, dma.get("bb_bandwidth"), compression_log)

        row = {
            "symbol": symbol,
            "last_close": dma.get("last_close"),
            "dma_signal": dma.get("signal"),
            "rs_20d": dma.get("rs_20d"),
            "rs_60d": dma.get("rs_60d"),
            "adx14": dma.get("adx14"),
            "vol_ratio_20d": dma.get("vol_ratio_20d"),
            "compression_percentile": compression["percentile"],
            "compression_days_of_history": compression["days_of_history"],
            "is_compressed": compression["is_compressed"],
            "delivery_zscore_20d": delivery.get("delivery_zscore_20d"),
            "delivery_signal": delivery.get("signal"),
            "known_investors_involved": smart_money.get("known_investors_involved", []),
            "repeat_buyers": smart_money.get("repeat_buyers", []),
            "smart_money_net_value_cr": smart_money.get("net_value_cr"),
        }
        row["score"] = score_row(row)
        row["stage"] = classify_stage(row)
        results.append(row)

    results.sort(key=lambda r: r["score"], reverse=True)

    symbols_with_enough_compression_history = sum(1 for r in results if r["compression_days_of_history"] >= 10)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(symbols),
        "success_count": len(results),
        "symbols_with_compression_history": symbols_with_enough_compression_history,
        "compression_log_days_total": int(compression_log["date"].nunique()) if not compression_log.empty else 0,
        "methodology_note": "Built from real, documented case-study research (Paras Defence specifically), NOT a "
                             "statistically backtested model. Read the README before trusting this tool's ranking "
                             "the way you'd trust DMA Radar's or Delivery Radar's.",
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s (%d results, %d with meaningful compression history)",
              args.out, len(results), symbols_with_enough_compression_history)


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def demo_dma_context(symbols: list[str]) -> dict[str, dict]:
    rng = np.random.default_rng(77)
    out = {}
    for s in symbols:
        out[s] = {
            "last_close": round(float(rng.uniform(80, 3000)), 2),
            "signal": rng.choice(["no_signal", "approaching_golden_cross", "confirmed_golden_cross", "no_signal"]),
            "rs_20d": round(float(rng.normal(3, 15)), 2),
            "rs_60d": round(float(rng.normal(5, 20)), 2),
            "adx14": round(float(rng.uniform(10, 40)), 1),
            "vol_ratio_20d": round(float(rng.uniform(0.6, 2.2)), 2),
            "bb_bandwidth": round(float(rng.uniform(0.02, 0.18)), 4),
        }
    return out


def demo_delivery_context(symbols: list[str]) -> dict[str, dict]:
    rng = np.random.default_rng(78)
    out = {}
    for s in symbols:
        out[s] = {
            "delivery_zscore_20d": round(float(rng.normal(0.3, 1.2)), 2),
            "signal": rng.choice(["no_signal", "quiet_accumulation", "no_signal", "no_signal"]),
        }
    return out


def demo_smart_money_context(symbols: list[str]) -> dict[str, dict]:
    rng = np.random.default_rng(79)
    out = {}
    for s in symbols:
        has_known = rng.random() > 0.9
        out[s] = {
            "known_investors_involved": ["DEMO INVESTOR"] if has_known else [],
            "repeat_buyers": ["Demo HNI"] if rng.random() > 0.85 else [],
            "net_value_cr": round(float(rng.normal(0, 20)), 2),
        }
    return out


if __name__ == "__main__":
    main()
