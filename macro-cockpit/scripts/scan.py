#!/usr/bin/env python3
"""
FII/DII + Participant-wise OI Macro Cockpit
------------------------------------------------
One "should I be aggressive today" gauge, checked before looking at any
single stock. Four pieces:

    1. India VIX (with trend) - the fear gauge
    2. Advance-Decline (Nifty 50 and Nifty 500) - market breadth
    3. Participant-wise F&O open interest (FII/DII/Pro/Client net index
       futures positioning) - who's leaning which way
    4. FII/DII cash market net flow - the other half of "smart money"

DATA SOURCE CONFIDENCE, so you know where to look first if something's
off:
    - VIX: HIGH confidence. Official NSE historical VIX endpoint via the
      `nse` package, supports a real date range - backfills automatically.
    - Advance-Decline: HIGH confidence. Official NSE endpoint via `nse`,
      but only returns TODAY's snapshot (no historical range), so the
      5-day/20-day trend builds up from a local cache over time rather
      than backfilling immediately.
    - Participant OI: HIGH confidence. This is a plain daily CSV file NSE
      publishes openly (same reliable pattern as Delivery Radar's
      bhavcopy) - backfills like Delivery Radar does.
    - FII/DII cash flow: LOWER confidence. The endpoint exists and is
      well-documented in the open-source NSE-scraping community, but its
      exact response shape could not be verified from a fully sandboxed
      build environment. Parsing is written defensively (tries several
      plausible field names, logs clearly and degrades gracefully rather
      than crashing if the shape doesn't match) - this is the one piece
      most likely to need a follow-up adjustment once you see real output.

Usage:
    python scan.py --demo     # synthetic data, no network calls
    python scan.py            # live, incremental cache update
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
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("macro-cockpit")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "scan_results.json"
HISTORY_CACHE_PATH = ROOT / "data" / "macro_history.csv"
PARTICIPANT_CACHE_PATH = ROOT / "data" / "participant_oi_cache.csv"
NSE_CACHE_DIR = ROOT / ".nse_cache"

VIX_BACKFILL_DAYS = 40          # comfortably covers a 20-day trend with buffer for holidays
PARTICIPANT_BACKFILL_DAYS = 40
CACHE_RETENTION_DAYS = 400      # keep just over a year

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

HISTORY_COLUMNS = [
    "date", "vix_close", "nifty50_advances", "nifty50_declines",
    "nifty500_advances", "nifty500_declines", "fii_cash_net_cr", "dii_cash_net_cr",
]
PARTICIPANT_COLUMNS = ["date", "participant", "index_fut_long", "index_fut_short", "index_fut_net"]


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


# ---------------------------------------------------------------------------
# Generic cache helpers (same pattern as delivery-radar / smart-money)
# ---------------------------------------------------------------------------

def load_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=columns)


def save_cache(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(path, index=False)


def trim_cache(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=CACHE_RETENTION_DAYS))
    return df[df["date"] >= cutoff]


# ---------------------------------------------------------------------------
# VIX (high confidence - real historical range supported)
# ---------------------------------------------------------------------------

def fetch_vix_live(days: int) -> pd.DataFrame:
    from nse import NSE

    NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days)
    with NSE(str(NSE_CACHE_DIR), server=True) as nse:
        try:
            raw = nse.fetch_historical_vix_data(from_date=from_date.date(), to_date=to_date.date())
        except Exception as exc:  # noqa: BLE001
            log.warning("VIX fetch failed: %s", exc)
            return pd.DataFrame(columns=["date", "vix_close"])

    rows = []
    for r in raw:
        try:
            d = pd.to_datetime(r["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
            if pd.notna(d):
                rows.append({"date": d, "vix_close": float(r["EOD_CLOSE_INDEX_VAL"])})
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(rows)


def demo_vix(days: int) -> pd.DataFrame:
    rng = np.random.default_rng(10)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days + 5)[-days:]  # buffer+slice, see dma-radar/scripts/scan.py for why
    vix = np.clip(rng.normal(13, 2, len(dates)).cumsum() * 0.05 + 12, 8, 35)
    return pd.DataFrame({"date": dates, "vix_close": vix})


# ---------------------------------------------------------------------------
# Advance-Decline (high confidence, today-only - accumulates via cache)
# ---------------------------------------------------------------------------

def fetch_advance_decline_live(index: str) -> dict | None:
    from nse import NSE

    NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with NSE(str(NSE_CACHE_DIR), server=True) as nse:
        try:
            return nse.advanceDecline(index)
        except Exception as exc:  # noqa: BLE001
            log.warning("Advance-decline fetch failed for %s: %s", index, exc)
            return None


def demo_advance_decline(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    adv = int(rng.integers(15, 40))
    total = 50 if seed == 20 else int(rng.integers(300, 500))
    return {"advances": str(adv), "declines": str(max(total - adv, 0)), "unchanged": "0"}


# ---------------------------------------------------------------------------
# Participant-wise Open Interest (high confidence - plain daily CSV)
# ---------------------------------------------------------------------------

def participant_oi_url(date: datetime) -> str:
    return f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{date.strftime('%d%m%Y')}.csv"


def fetch_participant_oi_day_live(date: datetime) -> pd.DataFrame | None:
    try:
        resp = requests.get(participant_oi_url(date), headers=HEADERS, timeout=20)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        # This file has a title line before the real header row
        lines = resp.text.strip().splitlines()
        header_idx = next((i for i, l in enumerate(lines) if l.strip().lower().startswith("client type")), None)
        if header_idx is None:
            return None
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as exc:  # noqa: BLE001
        log.debug("Participant OI fetch failed for %s: %s", date.date(), exc)
        return None


def normalize_participant_oi(raw: pd.DataFrame, date: datetime) -> list[dict]:
    long_col = find_col(raw.columns, ["futureindexlong"])
    short_col = find_col(raw.columns, ["futureindexshort"])
    type_col = find_col(raw.columns, ["clienttype"])
    if not all([long_col, short_col, type_col]):
        return []

    rows = []
    for _, r in raw.iterrows():
        participant = str(r[type_col]).strip().upper()
        if participant not in ("CLIENT", "DII", "FII", "PRO"):
            continue
        try:
            long_c = float(str(r[long_col]).replace(",", ""))
            short_c = float(str(r[short_col]).replace(",", ""))
        except (ValueError, TypeError):
            continue
        rows.append({
            "date": date, "participant": participant,
            "index_fut_long": long_c, "index_fut_short": short_c,
            "index_fut_net": long_c - short_c,
        })
    return rows


def demo_participant_oi_day(date: datetime, seed: int) -> list[dict]:
    rng = np.random.default_rng(abs(hash((date.date(), seed))) % (2**32))
    base = {"CLIENT": 50000, "DII": 5000, "FII": 40000, "PRO": 60000}
    rows = []
    for p, b in base.items():
        long_c = b + rng.integers(-8000, 8000)
        short_c = b + rng.integers(-8000, 8000)
        rows.append({"date": date, "participant": p, "index_fut_long": long_c,
                      "index_fut_short": short_c, "index_fut_net": long_c - short_c})
    return rows


def update_participant_cache(demo: bool) -> pd.DataFrame:
    cache = load_cache(PARTICIPANT_CACHE_PATH, PARTICIPANT_COLUMNS)
    have_dates = set(cache["date"].dt.date) if not cache.empty else set()
    today = datetime.now(timezone.utc).date()
    lookback = PARTICIPANT_BACKFILL_DAYS if cache.empty else 7

    new_rows = []
    for i in range(1, lookback + 1):
        d = datetime.combine(today, datetime.min.time()) - timedelta(days=i)
        if d.date() in have_dates or d.weekday() >= 5:
            continue
        if demo:
            new_rows.extend(demo_participant_oi_day(d, seed=1))
        else:
            raw = fetch_participant_oi_day_live(d)
            if raw is not None:
                new_rows.extend(normalize_participant_oi(raw, d))

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates(subset=["date", "participant"], keep="last")
    cache = trim_cache(cache)
    save_cache(cache, PARTICIPANT_CACHE_PATH)
    return cache


# ---------------------------------------------------------------------------
# FII/DII cash flow (LOWER confidence - defensive parsing, see module docstring)
# ---------------------------------------------------------------------------

def fetch_fii_dii_cash_live() -> dict | None:
    from nse import NSE

    NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with NSE(str(NSE_CACHE_DIR), server=True) as nse:
        try:
            resp = nse._req("https://www.nseindia.com/api/fiidiiTradeReact")
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("FII/DII cash fetch failed: %s", exc)
            return None


def parse_fii_dii_cash(raw) -> dict:
    """Defensive parsing - the exact response shape wasn't verifiable
    ahead of time (see module docstring). Tries several plausible field
    name variants; returns Nones rather than crashing if nothing matches,
    and logs the raw shape so a real failure is easy to diagnose."""
    result = {"fii_cash_net_cr": None, "dii_cash_net_cr": None}
    if not raw or not isinstance(raw, list):
        log.warning("FII/DII cash: unexpected response shape (not a list): %s", str(raw)[:200])
        return result

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cat_key = find_col(entry.keys(), ["category"])
        net_key = find_col(entry.keys(), ["netvalue", "net"])
        if cat_key is None or net_key is None:
            continue
        category = str(entry[cat_key]).upper()
        try:
            net_val = float(str(entry[net_key]).replace(",", ""))
        except (ValueError, TypeError):
            continue
        if "FII" in category or "FPI" in category:
            result["fii_cash_net_cr"] = net_val
        elif "DII" in category:
            result["dii_cash_net_cr"] = net_val

    if result["fii_cash_net_cr"] is None and result["dii_cash_net_cr"] is None:
        log.warning("FII/DII cash: parsed 0 of 2 values from response - field names may have changed. "
                    "Raw sample: %s", json.dumps(raw)[:300] if raw else "empty")
    return result


def demo_fii_dii_cash(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"fii_cash_net_cr": round(float(rng.normal(-200, 1500)), 2),
            "dii_cash_net_cr": round(float(rng.normal(400, 1200)), 2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_trend(series: pd.Series, window: int) -> float | None:
    valid = series.dropna()
    if len(valid) < min(window, 3):
        return None
    return round(float(valid.tail(window).mean()), 2)


# ---------------------------------------------------------------------------
# Signal Breadth: a genuinely new macro indicator, built entirely from data
# DMA Radar and Delivery Radar already produce (their signal_log.csv files,
# sibling reads, zero extra NSE calls). Nothing here talks to NSE at all -
# it's the ecosystem's own daily output, aggregated into a market-wide
# breadth read no single per-stock tool could show on its own. This is
# the clearest example of "interconnected web" in this whole system: two
# tools built for per-stock analysis become, in aggregate, a market gauge.
# ---------------------------------------------------------------------------

DMA_BULLISH_SIGNALS = {"confirmed_golden_cross", "approaching_golden_cross"}
DMA_BEARISH_SIGNALS = {"confirmed_death_cross", "approaching_death_cross"}
DELIVERY_BULLISH_SIGNALS = {"quiet_accumulation", "confirmed_accumulation"}
DELIVERY_BEARISH_SIGNALS = {"possible_distribution"}


def _load_sibling_signal_log(tool_folder: str) -> pd.DataFrame | None:
    path = ROOT.parent / tool_folder / "data" / "signal_log.csv"
    if not path.exists():
        log.info("No signal_log.csv found for %s at %s - Signal Breadth will exclude it this run.", tool_folder, path)
        return None
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s signal log: %s", tool_folder, exc)
        return None


def compute_signal_breadth() -> dict:
    dma_log = _load_sibling_signal_log("dma-radar")
    delivery_log = _load_sibling_signal_log("delivery-radar")

    daily_rows = []
    for df, bullish_set, bearish_set, label in (
        (dma_log, DMA_BULLISH_SIGNALS, DMA_BEARISH_SIGNALS, "dma"),
        (delivery_log, DELIVERY_BULLISH_SIGNALS, DELIVERY_BEARISH_SIGNALS, "delivery"),
    ):
        if df is None or df.empty:
            continue
        grouped = df.groupby("date")["signal"].agg(
            bullish=lambda s: s.isin(bullish_set).sum(),
            bearish=lambda s: s.isin(bearish_set).sum(),
        ).reset_index()
        grouped["source"] = label
        daily_rows.append(grouped)

    if not daily_rows:
        return {"available": False, "note": "Neither DMA Radar nor Delivery Radar has a signal log yet - "
                                              "this fills in once both tools have run at least once."}

    combined = pd.concat(daily_rows, ignore_index=True)
    by_date = combined.groupby("date")[["bullish", "bearish"]].sum().reset_index()
    by_date["net"] = by_date["bullish"] - by_date["bearish"]
    by_date = by_date.sort_values("date")

    today_row = by_date.iloc[-1] if not by_date.empty else None
    history_20d = by_date.tail(21).iloc[:-1] if len(by_date) > 1 else by_date.iloc[0:0]  # exclude today from its own baseline

    return {
        "available": True,
        "today_bullish": int(today_row["bullish"]) if today_row is not None else None,
        "today_bearish": int(today_row["bearish"]) if today_row is not None else None,
        "today_net": int(today_row["net"]) if today_row is not None else None,
        "avg_net_20d": round(float(history_20d["net"].mean()), 1) if len(history_20d) >= 3 else None,
        "days_of_history": len(by_date),
        "history": [
            {"date": r["date"].strftime("%Y-%m-%d"), "bullish": int(r["bullish"]), "bearish": int(r["bearish"]), "net": int(r["net"])}
            for _, r in by_date.tail(30).iterrows()
        ],
    }


def demo_signal_breadth() -> dict:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=30)[-25:]  # buffer+slice, see dma-radar/scripts/scan.py for why
    history = []
    for d in dates:
        bullish = int(rng.integers(3, 18))
        bearish = int(rng.integers(2, 15))
        history.append({"date": d.strftime("%Y-%m-%d"), "bullish": bullish, "bearish": bearish, "net": bullish - bearish})
    today = history[-1]
    avg_net_20d = round(sum(h["net"] for h in history[-21:-1]) / 20, 1)
    return {
        "available": True, "today_bullish": today["bullish"], "today_bearish": today["bearish"],
        "today_net": today["net"], "avg_net_20d": avg_net_20d, "days_of_history": len(history),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="FII/DII + Participant OI Macro Cockpit")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    today = pd.Timestamp(datetime.now(timezone.utc).date())

    # --- VIX: backfill-capable, always refresh the full window ---
    vix_df = demo_vix(VIX_BACKFILL_DAYS) if args.demo else fetch_vix_live(VIX_BACKFILL_DAYS)
    log.info("VIX: %d day(s) fetched", len(vix_df))

    # --- Advance-decline: today-only, load+append to history cache ---
    history = load_cache(HISTORY_CACHE_PATH, HISTORY_COLUMNS)
    ad50 = demo_advance_decline(20) if args.demo else fetch_advance_decline_live("NIFTY 50")
    ad500 = demo_advance_decline(21) if args.demo else fetch_advance_decline_live("NIFTY 500")
    fii_dii = demo_fii_dii_cash(22) if args.demo else parse_fii_dii_cash(fetch_fii_dii_cash_live())

    today_row = {
        "date": today,
        "vix_close": float(vix_df[vix_df["date"] == today]["vix_close"].iloc[0]) if not vix_df.empty and (vix_df["date"] == today).any() else (float(vix_df["vix_close"].iloc[-1]) if not vix_df.empty else None),
        "nifty50_advances": int(ad50["advances"]) if ad50 else None,
        "nifty50_declines": int(ad50["declines"]) if ad50 else None,
        "nifty500_advances": int(ad500["advances"]) if ad500 else None,
        "nifty500_declines": int(ad500["declines"]) if ad500 else None,
        "fii_cash_net_cr": fii_dii.get("fii_cash_net_cr"),
        "dii_cash_net_cr": fii_dii.get("dii_cash_net_cr"),
    }

    if not history.empty and (history["date"] == today).any():
        for k, v in today_row.items():
            if k != "date" and v is not None:
                history.loc[history["date"] == today, k] = v
    else:
        history = pd.concat([history, pd.DataFrame([today_row])], ignore_index=True)

    # Backfill VIX into history for any past days we now have but didn't before
    for _, r in vix_df.iterrows():
        mask = history["date"] == r["date"]
        if mask.any():
            if pd.isna(history.loc[mask, "vix_close"]).all():
                history.loc[mask, "vix_close"] = r["vix_close"]
        else:
            new_r = {c: None for c in HISTORY_COLUMNS}
            new_r["date"] = r["date"]
            new_r["vix_close"] = r["vix_close"]
            history = pd.concat([history, pd.DataFrame([new_r])], ignore_index=True)

    history = history.drop_duplicates(subset=["date"], keep="last")
    history = trim_cache(history)
    save_cache(history, HISTORY_CACHE_PATH)
    log.info("Macro history cache: %d day(s) total", len(history))

    # --- Participant OI: incremental cache, same pattern as delivery-radar ---
    participant_cache = update_participant_cache(demo=args.demo)
    log.info("Participant OI cache: %d row(s) total", len(participant_cache))

    # --- Assemble output ---
    history_sorted = history.sort_values("date")
    latest = history_sorted.iloc[-1] if not history_sorted.empty else None

    vix_trend = {
        "latest": round(float(latest["vix_close"]), 2) if latest is not None and pd.notna(latest["vix_close"]) else None,
        "avg_5d": compute_trend(history_sorted["vix_close"], 5),
        "avg_20d": compute_trend(history_sorted["vix_close"], 20),
    }

    breadth = {
        "nifty50": {"advances": int(latest["nifty50_advances"]) if latest is not None and pd.notna(latest["nifty50_advances"]) else None,
                     "declines": int(latest["nifty50_declines"]) if latest is not None and pd.notna(latest["nifty50_declines"]) else None},
        "nifty500": {"advances": int(latest["nifty500_advances"]) if latest is not None and pd.notna(latest["nifty500_advances"]) else None,
                      "declines": int(latest["nifty500_declines"]) if latest is not None and pd.notna(latest["nifty500_declines"]) else None},
    }

    fii_dii_series = history_sorted[["date", "fii_cash_net_cr", "dii_cash_net_cr"]].copy()
    fii_dii_trend = {
        "fii_latest": round(float(latest["fii_cash_net_cr"]), 2) if latest is not None and pd.notna(latest["fii_cash_net_cr"]) else None,
        "dii_latest": round(float(latest["dii_cash_net_cr"]), 2) if latest is not None and pd.notna(latest["dii_cash_net_cr"]) else None,
        "fii_avg_5d": compute_trend(history_sorted["fii_cash_net_cr"], 5),
        "fii_avg_20d": compute_trend(history_sorted["fii_cash_net_cr"], 20),
        "dii_avg_5d": compute_trend(history_sorted["dii_cash_net_cr"], 5),
        "dii_avg_20d": compute_trend(history_sorted["dii_cash_net_cr"], 20),
        "history": [
            {"date": r["date"].strftime("%Y-%m-%d"), "fii": r["fii_cash_net_cr"] if pd.notna(r["fii_cash_net_cr"]) else None,
             "dii": r["dii_cash_net_cr"] if pd.notna(r["dii_cash_net_cr"]) else None}
            for _, r in fii_dii_series.tail(40).iterrows()
        ],
    }

    vix_history_out = [
        {"date": r["date"].strftime("%Y-%m-%d"), "close": round(float(r["vix_close"]), 2)}
        for _, r in history_sorted.tail(40).iterrows() if pd.notna(r["vix_close"])
    ]

    participant_latest_date = participant_cache["date"].max() if not participant_cache.empty else None
    participant_summary = []
    if participant_latest_date is not None:
        latest_p = participant_cache[participant_cache["date"] == participant_latest_date]
        prior_dates = sorted(participant_cache["date"].unique())
        prior_date = prior_dates[-2] if len(prior_dates) >= 2 else None
        prior_p = participant_cache[participant_cache["date"] == prior_date] if prior_date is not None else pd.DataFrame()

        for participant in ("FII", "DII", "PRO", "CLIENT"):
            row = latest_p[latest_p["participant"] == participant]
            if row.empty:
                continue
            net_today = float(row["index_fut_net"].iloc[0])
            change = None
            if not prior_p.empty:
                prow = prior_p[prior_p["participant"] == participant]
                if not prow.empty:
                    change = round(net_today - float(prow["index_fut_net"].iloc[0]), 0)
            participant_summary.append({
                "participant": participant, "index_fut_net": round(net_today, 0),
                "change_vs_prior_day": change,
            })

    signal_breadth = demo_signal_breadth() if args.demo else compute_signal_breadth()
    if signal_breadth.get("available"):
        log.info("Signal breadth: today net=%s (bullish=%s, bearish=%s), 20d avg=%s, %d day(s) of history",
                  signal_breadth["today_net"], signal_breadth["today_bullish"], signal_breadth["today_bearish"],
                  signal_breadth["avg_net_20d"], signal_breadth["days_of_history"])

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vix": vix_trend,
        "vix_history_40d": vix_history_out,
        "vix_as_of": str(vix_df["date"].max().date()) if not vix_df.empty else None,  # VIX has real per-day dates - genuinely verifiable
        "breadth": breadth,
        "breadth_date_confidence": "unverified - NSE's advance-decline endpoint returns a live snapshot with no date field attached, "
                                    "so there's no way to independently confirm this reflects today's session rather than a stale cached "
                                    "value. Cross-check against vix_as_of below - if that's not today, treat this the same way.",
        "fii_dii_cash": fii_dii_trend,
        "fii_dii_date_confidence": "same caveat as breadth_date_confidence - this endpoint has no date field either",
        "participant_oi": {
            "as_of": participant_latest_date.strftime("%Y-%m-%d") if participant_latest_date is not None else None,  # real dated file - genuinely verifiable
            "index_futures_positioning": participant_summary,
        },
        "signal_breadth": signal_breadth,
        "history_days_cached": len(history),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
