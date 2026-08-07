#!/usr/bin/env python3
"""
Options PCR / Max Pain / OI Heatmap
--------------------------------------
A live-ish snapshot of Nifty and Bank Nifty options positioning: PCR
(put-call ratio), max pain, and where OI is concentrated by strike -
support/resistance sourced from actual positioning, not chart lines.

WHY THIS TOOL IS ARCHITECTURALLY DIFFERENT from everything else in this
repo: every other scanner here is once-a-day (EOD). Options positioning
changes throughout the trading session, so this one is meant to be
triggered every 5-10 minutes DURING market hours (9:15 AM - 3:30 PM IST) -
via the external cron-job.org setup, not GitHub's own `schedule:` trigger
(which has neither the granularity nor, as this repo's history shows, the
reliability for that cadence).

Each run is a fast, stateless snapshot for the "right now" view - but it
also appends a lightweight row to data/intraday_history.csv, so the
dashboard can show how PCR and max pain moved through TODAY's session.
That history resets each day; there's no multi-day backfill here, since
"how did today's positioning build up" is the point, not multi-day trend.

Uses `nse.compileOptionChain()`, which does the real math (max pain,
per-strike PCR, max Call/Put OI strikes) internally - this script mostly
just calls it, trims the result to strikes near the money for a readable
heatmap, and manages the intraday history file.

Usage:
    python scan.py --demo     # synthetic data, no network calls
    python scan.py            # live snapshot + append to today's history
"""

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("options-radar")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "scan_results.json"
HISTORY_PATH = ROOT / "data" / "intraday_history.csv"
NSE_CACHE_DIR = ROOT / ".nse_cache"

IST = timezone(timedelta(hours=5, minutes=30))
SYMBOLS = ["nifty", "banknifty"]
STRIKES_AROUND_ATM = 12   # how many strikes on each side of ATM to keep for the heatmap
HISTORY_COLUMNS = ["date", "time", "symbol", "underlying", "pcr", "maxpain", "atm"]


def now_ist() -> datetime:
    return datetime.now(timezone.utc).astimezone(IST)


# ---------------------------------------------------------------------------
# Live fetch
# ---------------------------------------------------------------------------

def fetch_symbol_live(symbol: str) -> dict | None:
    from nse import NSE

    NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with NSE(str(NSE_CACHE_DIR), server=True) as nse:
        try:
            raw = nse.optionChain(symbol)
            expiry_strs = raw.get("records", {}).get("expiryDates", [])
            if not expiry_strs:
                log.warning("%s: no expiry dates in response", symbol)
                return None
            nearest_expiry = datetime.strptime(expiry_strs[0], "%d-%b-%Y")
            compiled = nse.compileOptionChain(symbol, nearest_expiry)
            return compiled
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: fetch failed: %s", symbol, exc)
            return None


def demo_symbol(symbol: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base = 24500 if symbol == "nifty" else 51500
    step = 50 if symbol == "nifty" else 100
    underlying = base + rng.integers(-200, 200)
    atm = step * round(underlying / step)

    chain = {}
    total_coi = total_poi = 0
    max_coi = max_poi = 0
    max_coi_strike = max_poi_strike = atm
    for offset in range(-15, 16):
        strike = atm + offset * step
        # OI naturally concentrates near ATM and at round "psychological" strikes
        weight = np.exp(-abs(offset) / 8) * (1.6 if offset % 5 == 0 else 1.0)
        coi = max(0, int(rng.normal(500000, 150000) * weight))
        poi = max(0, int(rng.normal(480000, 150000) * weight * 1.05))
        chain[str(strike)] = {
            "ce": {"last": round(float(rng.uniform(5, 400)), 2), "oi": coi, "chg": int(rng.integers(-50000, 50000)), "iv": round(float(rng.uniform(11, 18)), 2)},
            "pe": {"last": round(float(rng.uniform(5, 400)), 2), "oi": poi, "chg": int(rng.integers(-50000, 50000)), "iv": round(float(rng.uniform(11, 18)), 2)},
            "pcr": round(poi / coi, 2) if coi else None,
        }
        total_coi += coi
        total_poi += poi
        if coi > max_coi:
            max_coi, max_coi_strike = coi, strike
        if poi > max_poi:
            max_poi, max_poi_strike = poi, strike

    return {
        "expiry": (now_ist() + timedelta(days=(3 - now_ist().weekday()) % 7 + 1)).strftime("%d-%b-%Y"),
        "timestamp": now_ist().strftime("%d-%b-%Y %H:%M:%S"),
        "underlying": float(underlying), "atm": atm,
        "maxpain": atm + step * int(rng.integers(-3, 3)),
        "maxCoi": max_coi_strike, "maxPoi": max_poi_strike,
        "coiTotal": total_coi, "poiTotal": total_poi,
        "pcr": round(total_poi / total_coi, 2) if total_coi else None,
        "chain": chain,
    }


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def trim_chain_to_heatmap(compiled: dict) -> list[dict]:
    """Keeps only strikes near the money, sorted, for a readable heatmap -
    the full chain often spans 100+ strikes where the far wings carry
    negligible OI and just add noise to a visual."""
    atm = compiled.get("atm")
    chain = compiled.get("chain", {})
    if atm is None or not chain:
        return []

    strikes = sorted(int(k) for k in chain.keys())
    if atm not in strikes:
        atm = min(strikes, key=lambda s: abs(s - atm))
    atm_idx = strikes.index(atm)
    window = strikes[max(0, atm_idx - STRIKES_AROUND_ATM):atm_idx + STRIKES_AROUND_ATM + 1]

    out = []
    for s in window:
        row = chain[str(s)]
        out.append({
            "strike": s, "is_atm": s == atm,
            "call_oi": row.get("ce", {}).get("oi", 0), "put_oi": row.get("pe", {}).get("oi", 0),
            "call_ltp": row.get("ce", {}).get("last", 0), "put_ltp": row.get("pe", {}).get("last", 0),
            "call_iv": row.get("ce", {}).get("iv", 0), "put_iv": row.get("pe", {}).get("iv", 0),
        })
    return out


def update_intraday_history(snapshot_rows: list[dict], demo: bool) -> pd.DataFrame:
    if HISTORY_PATH.exists():
        cache = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    else:
        cache = pd.DataFrame(columns=HISTORY_COLUMNS)

    today = pd.Timestamp(now_ist().date())
    cache = cache[cache["date"] == today]  # today-only history, by design

    new_df = pd.DataFrame(snapshot_rows)
    cache = pd.concat([cache, new_df], ignore_index=True)
    cache = cache.drop_duplicates(subset=["date", "time", "symbol"], keep="last")

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache.sort_values(["symbol", "time"]).to_csv(HISTORY_PATH, index=False)
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Options PCR / Max Pain / OI Heatmap")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    ist_now = now_ist()
    results = {}
    snapshot_rows = []

    for i, symbol in enumerate(SYMBOLS):
        compiled = demo_symbol(symbol, seed=30 + i) if args.demo else fetch_symbol_live(symbol)
        if compiled is None:
            log.warning("%s: no data this run", symbol)
            continue

        heatmap = trim_chain_to_heatmap(compiled)
        results[symbol] = {
            "expiry": compiled.get("expiry"),
            "underlying": compiled.get("underlying"),
            "atm": compiled.get("atm"),
            "maxpain": compiled.get("maxpain"),
            "max_call_oi_strike": compiled.get("maxCoi"),
            "max_put_oi_strike": compiled.get("maxPoi"),
            "total_call_oi": compiled.get("coiTotal"),
            "total_put_oi": compiled.get("poiTotal"),
            "pcr": compiled.get("pcr"),
            "heatmap": heatmap,
        }
        log.info("%-10s underlying=%-9s atm=%-7s maxpain=%-7s pcr=%s",
                  symbol, results[symbol]["underlying"], results[symbol]["atm"],
                  results[symbol]["maxpain"], results[symbol]["pcr"])

        snapshot_rows.append({
            "date": pd.Timestamp(ist_now.date()), "time": ist_now.strftime("%H:%M"),
            "symbol": symbol, "underlying": compiled.get("underlying"),
            "pcr": compiled.get("pcr"), "maxpain": compiled.get("maxpain"), "atm": compiled.get("atm"),
        })

    history = update_intraday_history(snapshot_rows, demo=args.demo) if snapshot_rows else pd.DataFrame(columns=HISTORY_COLUMNS)

    intraday_out = {}
    for symbol in SYMBOLS:
        sub = history[history["symbol"] == symbol].sort_values("time") if not history.empty else pd.DataFrame()
        intraday_out[symbol] = [
            {"time": r["time"], "pcr": r["pcr"], "maxpain": r["maxpain"], "underlying": r["underlying"]}
            for _, r in sub.iterrows()
        ]

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_at_ist": ist_now.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "intraday_today": intraday_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
