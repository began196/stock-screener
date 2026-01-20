#!/usr/bin/env python
# coding: utf-8

"""
Value Investing Screener for S&P 500 and STOXX Europe 600 (public data)

What it does:
- Scrapes S&P 500 tickers from Wikipedia, and STOXX Europe 600 from iShares
- Pulls public market + fundamentals data via yfinance (Yahoo Finance)
- Computes quality metrics
- Estimates intrinsic value band using:
    1) DCF-lite (FCF-based, bear/base/bull)
    2) Peer EV/EBITDA fair value (median of universe)
- Produces BUY/HOLD/SELL signals with margin-of-safety rules
- Exports results to CSV

Notes / limitations:
- Yahoo fundamentals can be missing/spotty for some tickers; script flags missing fields.
- Peer EV/EBITDA is cross-sectional (not historical mean reversion).
- Interest coverage & ROIC are approximate because statement line-items vary across companies.
"""

# Imports
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Optional, List
import argparse

import numpy as np
import pandas as pd
import polars as pl
import requests
import yfinance as yf
import datetime

# Hyperparameter configuration

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
ISHARES_EXSA_HOLDINGS_CSV = (
    "https://www.ishares.com/ch/individual/en/products/251931/"
    "ishares-stoxx-europe-600-ucits-etf-de-fund/1495092304805.ajax"
    "?dataType=fund&fileName=EXSA_holdings&fileType=csv"
)

@dataclass
class DCFParams:
    years: int = 5
    discount_rate: float = 0.10
    terminal_growth: float = 0.02
    growth_bear: float = 0.00
    growth_base: float = 0.05
    growth_bull: float = 0.10

@dataclass
class SignalParams:
    margin_of_safety: float = 0.30
    sell_buffer: float = 0.15


# Define helper functions

def safe_get(d: Dict, key: str) -> Optional[float]:
    v = d.get(key, None)
    try:
        if v is None:
            return None
        if isinstance(v, (int, float, np.number)) and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
        return float(v)
    except Exception:
        return None

# Fetch SnP500 tickers
def fetch_sp500_tickers() -> pl.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=10)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))  # avoids FutureWarning
    sp500 = tables[0].copy()

    sp500.rename(columns={"Symbol": "Ticker"}, inplace=True)
    sp500["Ticker"] = sp500["Ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B

    return pl.from_pandas(sp500).with_columns([
        pl.lit("United States").alias("Country"),
        pl.lit("US").alias("Region"),
    ])

# Fetch STOXX Europe 600 tickers
def read_ishares_holdings_csv(text: str) -> pl.DataFrame:
    lines = text.splitlines()

    # 1) Find the header row (where the actual table begins)
    header_idx = None
    for i, line in enumerate(lines):
        if "Ticker" in line and "Name" in line:
            header_idx = i
            break
    if header_idx is None:
        # Helpful debug: show first ~30 lines so you can see what's in there
        preview = "\n".join(lines[:30])
        raise ValueError(
            "Could not find holdings table header row. "
            "Here are the first 30 lines:\n\n" + preview
        )

    # 2) Detect delimiter: iShares files are sometimes comma, sometimes semicolon
    header_line = lines[header_idx]
    sep = ";" if header_line.count(";") > header_line.count(",") else ","

    # 3) Parse from header row onward
    df = pd.read_csv(
        StringIO("\n".join(lines[header_idx:])),
        sep=sep,
        engine="python",       # more forgiving for odd quoting
    )

    return df

def fetch_stoxx600_from_ishares() -> pl.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(ISHARES_EXSA_HOLDINGS_CSV, headers=headers, timeout=20)
    r.raise_for_status()

    df = read_ishares_holdings_csv(r.text)

    # Standardize columns (guard against slight naming differences)
    # Common variants: "Ticker", "Issuer Ticker", "Issuer Ticker " 
    required = ["Ticker", "Name", "Sector", "Location"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing}. Found columns: {df.columns.tolist()}")

    out = pd.DataFrame({
        "Ticker": df["Ticker"].astype(str).str.strip(),
        "Security": df["Name"].astype(str).str.strip(),
        "GICS Sector": df["Sector"].astype(str).str.strip(),
        "GICS Sub-Industry": None,
        "Country": df["Location"].astype(str).str.strip(),
        "Exchange": df["Exchange"].astype(str).str.strip(),
        "Region": "Europe"
    })

    out = out[out["Ticker"].notna() & (out["Ticker"] != "")]
    out = out.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    out["Ticker"] = out["Ticker"].str.replace(".", "", regex=False)
    out["Ticker"] = out["Ticker"].str.replace(" ", "-", regex=False)

    out = pl.from_pandas(out)

    # Append exchange suffix
    exchange_suffix = {
        "Euronext Amsterdam": "AS",
        "SIX Swiss Exchange": "SW",
        "London Stock Exchange": "L",
        "Omx Nordic Exchange Copenhagen A/S": "CO",
        "Bolsa De Madrid": "MC",
        "Nyse Euronext - Euronext Paris": "PA",
        "Borsa Italiana": "MI",
        "Nyse Euronext - Euronext Brussels": "BE",
        "Nasdaq Omx Helsinki Ltd.": "HE",
        "Xetra": "DE"
    }
    out = out.with_columns(
        pl.when(pl.col("Exchange").is_in(exchange_suffix.keys()))
        .then(
            pl.col("Ticker") + "." + pl.col("Exchange").replace(exchange_suffix)
        )
        .otherwise(pl.col("Ticker"))
        .alias("Ticker")
    )

    return out

# Discounted cash flow and peer multiple valuation functions
def dcf_lite_equity_value_per_share(
    fcf: float, shares_out: float, net_debt: float, params: DCFParams, growth: float,
) -> Optional[float]:
    if fcf <= 0 or shares_out <= 0:
        return None
    r = params.discount_rate
    gT = params.terminal_growth
    if r <= gT:
        return None

    pv = 0.0
    fcf_t = fcf
    for t in range(1, params.years + 1):
        fcf_t *= (1.0 + growth)
        pv += fcf_t / ((1.0 + r) ** t)

    fcf_n1 = fcf_t * (1.0 + gT)
    tv = fcf_n1 / (r - gT)
    pv_tv = tv / ((1.0 + r) ** params.years)

    enterprise_value = pv + pv_tv
    equity_value = enterprise_value - net_debt
    return equity_value / shares_out

# Calculate Peer fair value
def peer_multiple_fair_value_per_share(
    peer_ev_to_ebitda_median: float, ebitda: float, shares_out: float, net_debt: float,
) -> Optional[float]:
    if peer_ev_to_ebitda_median <= 0 or ebitda <= 0 or shares_out <= 0:
        return None
    fair_ev = peer_ev_to_ebitda_median * ebitda
    fair_equity = fair_ev - net_debt
    return fair_equity / shares_out

# Calculate quality metrics
def compute_quality_metrics(info: Dict, ticker_obj: yf.Ticker) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}

    fcf = safe_get(info, "freeCashflow")
    ebitda = safe_get(info, "ebitda")
    shares_out = safe_get(info, "sharesOutstanding")

    total_debt = safe_get(info, "totalDebt")
    total_cash = safe_get(info, "totalCash")
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash
    elif total_debt is not None:
        net_debt = total_debt
    else:
        net_debt = None

    out["fcf"] = fcf
    out["ebitda"] = ebitda
    out["shares_out"] = shares_out
    out["net_debt"] = net_debt

    # Net Debt / EBITDA
    if net_debt is not None and ebitda is not None and ebitda > 0:
        out["net_debt_to_ebitda"] = net_debt / ebitda
    else:
        out["net_debt_to_ebitda"] = None

    # Interest coverage (best effort from annual financials)
    interest_expense = None
    ebit = None
    fin = None
    try:
        fin = ticker_obj.financials
        for k in ["Interest Expense", "InterestExpense"]:
            if k in fin.index:
                interest_expense = float(fin.loc[k].iloc[0])
                break
        for k in ["Ebit", "EBIT", "Operating Income", "OperatingIncome"]:
            if k in fin.index:
                ebit = float(fin.loc[k].iloc[0])
                break
    except Exception:
        pass

    if ebit is not None and interest_expense is not None and interest_expense != 0:
        out["interest_coverage"] = ebit / abs(interest_expense)
    else:
        out["interest_coverage"] = None

    # ROIC proxy (rough)
    tax_rate = None
    pretax = None
    tax_prov = None
    try:
        if fin is not None:
            if "Pretax Income" in fin.index:
                pretax = float(fin.loc["Pretax Income"].iloc[0])
            elif "PretaxIncome" in fin.index:
                pretax = float(fin.loc["PretaxIncome"].iloc[0])

            if "Tax Provision" in fin.index:
                tax_prov = float(fin.loc["Tax Provision"].iloc[0])
            elif "TaxProvision" in fin.index:
                tax_prov = float(fin.loc["TaxProvision"].iloc[0])

            if pretax not in (None, 0) and tax_prov is not None:
                tr = tax_prov / pretax
                tax_rate = float(np.clip(tr, 0.0, 0.40))
    except Exception:
        pass

    total_equity = safe_get(info, "totalStockholderEquity")
    if total_equity is not None and total_debt is not None:
        invested_capital = total_debt + total_equity - (total_cash or 0.0)
    else:
        invested_capital = None

    if ebit is not None and invested_capital is not None and invested_capital > 0:
        if tax_rate is None:
            tax_rate = 0.21
        nopat = ebit * (1.0 - tax_rate)
        out["roic_proxy"] = nopat / invested_capital
    else:
        out["roic_proxy"] = None

    return out

# Pull market snapshot
def build_universe_snapshot(tickers: List[str]) -> pl.DataFrame:
    rows = []

    for i, t in enumerate(tickers, 1):
        try:
            tk = yf.Ticker(t)
            info = tk.get_info()
            q = compute_quality_metrics(info, tk)

            row = {
                "ticker": t,
                "price": safe_get(info, "currentPrice"),
                "market_cap": safe_get(info, "marketCap"),
                "enterprise_value": safe_get(info, "enterpriseValue"),
                "ev_to_ebitda": safe_get(info, "enterpriseToEbitda"),
                "fcf": safe_get(info, "freeCashflow"),
                "ebitda": safe_get(info, "ebitda"),
                "shares_out": safe_get(info, "sharesOutstanding"),
                **q
            }

            td, tc = safe_get(info, "totalDebt"), safe_get(info, "totalCash")
            row["net_debt"] = td - tc if td and tc else td

            rows.append(row)
        except Exception as e:
            rows.append({"ticker": t, "error": str(e)})

        if i % 50 == 0:
            print(f"Fetched {i}/{len(tickers)} tickers...")

    return pl.DataFrame(rows)

# Quality screening 
def apply_quality_screen(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            # Positive cash generation / operating viability
            (pl.col("fcf").fill_null(-1) > 0).alias("qc_fcf_pos"),
            (pl.col("ebitda").fill_null(-1) > 0).alias("qc_ebitda_pos"),

            # Leverage: allow nulls (unknown) to pass, but fail if clearly too levered
            (pl.col("net_debt_to_ebitda").is_null() | (pl.col("net_debt_to_ebitda") <= 3.0)).alias("qc_leverage_ok"),

            # Debt service: allow nulls to pass
            (pl.col("interest_coverage").is_null() | (pl.col("interest_coverage") >= 3.0)).alias("qc_intcov_ok"),

            # ROIC proxy: allow nulls to pass
            (pl.col("roic_proxy").is_null() | (pl.col("roic_proxy") >= 0.08)).alias("qc_roic_ok"),
        ])
        .with_columns([
            (
                pl.col("qc_fcf_pos")
                & pl.col("qc_ebitda_pos")
                & pl.col("qc_leverage_ok")
                & pl.col("qc_intcov_ok")
                & pl.col("qc_roic_ok")
            ).alias("quality_pass")
        ])
    )


# Compute intrinsic values + signals
def compute_intrinsic_and_signals(
    df: pl.DataFrame,
    dcf_params: DCFParams,
    sig_params: SignalParams,
) -> pl.DataFrame:
    d = df.clone()

    # Peer EV/EBITDA median across universe (exclude null, inf, extreme)
    peer_median_row = (
        d
        .with_columns(pl.col("ev_to_ebitda").cast(pl.Float64))
        .filter(
            pl.col("ev_to_ebitda").is_not_null()
            & pl.col("ev_to_ebitda").is_finite()
            & (pl.col("ev_to_ebitda") > 0)
            & (pl.col("ev_to_ebitda") < 80)
        )
        .select(pl.col("ev_to_ebitda").median().alias("peer_median"))
    )

    peer_median = peer_median_row.item() if peer_median_row.height > 0 else None

    d = d.with_columns(pl.lit(peer_median).cast(pl.Float64).alias("peer_ev_to_ebitda_median"))

    dcf_bear: List[Optional[float]] = []
    dcf_base: List[Optional[float]] = []
    dcf_bull: List[Optional[float]] = []
    peer_fv: List[Optional[float]] = []

    for r in d.iter_rows(named=True):
        fcf = r.get("fcf")
        shares = r.get("shares_out")
        net_debt = r.get("net_debt")
        ebitda = r.get("ebitda")

        # DCF scenarios
        if fcf is None or shares is None or net_debt is None:
            dcf_bear.append(None)
            dcf_base.append(None)
            dcf_bull.append(None)
        else:
            dcf_bear.append(
                dcf_lite_equity_value_per_share(float(fcf), float(shares), float(net_debt), dcf_params, dcf_params.growth_bear)
            )
            dcf_base.append(
                dcf_lite_equity_value_per_share(float(fcf), float(shares), float(net_debt), dcf_params, dcf_params.growth_base)
            )
            dcf_bull.append(
                dcf_lite_equity_value_per_share(float(fcf), float(shares), float(net_debt), dcf_params, dcf_params.growth_bull)
            )

        # Peer multiple fair value
        if peer_median is None or ebitda is None or shares is None or net_debt is None:
            peer_fv.append(None)
        else:
            peer_fv.append(
                peer_multiple_fair_value_per_share(float(peer_median), float(ebitda), float(shares), float(net_debt))
            )

    d = d.with_columns([
        pl.Series("dcf_bear", dcf_bear, dtype=pl.Float64),
        pl.Series("dcf_base", dcf_base, dtype=pl.Float64),
        pl.Series("dcf_bull", dcf_bull, dtype=pl.Float64),
        pl.Series("peer_fair_value", peer_fv, dtype=pl.Float64),
    ])

    # Blend intrinsic estimate
    d = d.with_columns([
        pl.when(pl.col("dcf_base").is_not_null() & pl.col("peer_fair_value").is_not_null())
          .then(0.6 * pl.col("dcf_base") + 0.4 * pl.col("peer_fair_value"))
          .otherwise(pl.coalesce([pl.col("dcf_base"), pl.col("peer_fair_value")]))
          .alias("intrinsic_est")
    ])

    mos = float(sig_params.margin_of_safety)
    buf = float(sig_params.sell_buffer)

    d = d.with_columns([
        (pl.col("intrinsic_est") * (1.0 - mos)).alias("buy_threshold"),
        (pl.col("intrinsic_est") * (1.0 + buf)).alias("sell_threshold"),
    ])

    # Signal generation (fully vectorized)
    d = d.with_columns([
        pl.when(pl.col("price").is_null() | pl.col("intrinsic_est").is_null())
          .then(pl.lit("NO_DATA"))
          .when(~pl.col("quality_pass"))
          .then(
              pl.when(pl.col("price") <= pl.col("buy_threshold"))
                .then(pl.lit("CHEAP_BUT_FAILS_QUALITY"))
                .otherwise(pl.lit("FAILS_QUALITY"))
          )
          .when(pl.col("price") <= pl.col("buy_threshold"))
          .then(pl.lit("BUY"))
          .when(pl.col("price") >= pl.col("sell_threshold"))
          .then(pl.lit("SELL/TRIM"))
          .otherwise(pl.lit("HOLD"))
          .alias("signal")
    ])

    d = d.with_columns([
        (pl.col("intrinsic_est") / pl.col("price") - 1.0).alias("upside_to_intrinsic")
    ])

    return d

def main():

    # Define arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help = "If None, a timestamped csv file name will be used")
    ap.add_argument("--max", type=int, default=None, help="Limit tickers for quick tests (e.g., 50)")
    ap.add_argument("--discount", type=float, default=0.10)
    ap.add_argument("--terminal_growth", type=float, default=0.02)
    ap.add_argument("--mos", type=float, default=0.30)
    ap.add_argument("--sell_buffer", type=float, default=0.15)
    args = ap.parse_args()

    OUT_PATH = args.out
    MAX_TICKERS = args.max
    dcf_params = DCFParams(
        discount_rate=args.discount,
        terminal_growth=args.terminal_growth,
    )
    sig_params = SignalParams(
        margin_of_safety=args.mos,
        sell_buffer=args.sell_buffer,
    )

    sp500 = fetch_sp500_tickers()
    europe = fetch_stoxx600_from_ishares()

    # Combine both ticker DFs
    ticker_cols = ["Ticker", "Security", "Country", "Region", "GICS Sector", "GICS Sub-Industry"]
    universe = pl.concat([sp500.select(ticker_cols), europe.select(ticker_cols)], how="vertical")

    # Get ticker symbols
    snp500_tickers = sp500["Ticker"].to_list()
    europe_tickers = europe["Ticker"].to_list()
    if MAX_TICKERS is not None:
        snp500_tickers = snp500_tickers[:MAX_TICKERS]
        europe_tickers = europe_tickers[:MAX_TICKERS]

    tickers = snp500_tickers + europe_tickers

    print(f"Loaded {len(tickers)} S&P 500 + STOXX Europe 600 tickers.", file=sys.stderr)

    df = build_universe_snapshot(tickers)

    # Merge df with company details
    df = df.join(
        universe.select(["Ticker", "Security", "GICS Sector", "GICS Sub-Industry", "Country", "Region"]).rename({"Ticker": "ticker"}),
        on="ticker",
        how="left",
    )

    df = apply_quality_screen(df)

    # Calculate intrinsic value and generate signals based on dcf and signal hyperparameters
    df = compute_intrinsic_and_signals(df, dcf_params, sig_params)

    # Export data
    watchlist_cols = [
        "ticker", "Security", "GICS Sector", "GICS Sub-Industry", "Country", "Region",
        "price", "intrinsic_est", "buy_threshold", "sell_threshold",
        "upside_to_intrinsic", "signal", "quality_pass",
        "net_debt_to_ebitda", "interest_coverage", "roic_proxy",
        "ev_to_ebitda", "peer_ev_to_ebitda_median",
        "dcf_bear", "dcf_base", "dcf_bull", "peer_fair_value",
        "fcf", "ebitda", "net_debt", "shares_out",
    ]

    watchlist = df.select(watchlist_cols)

    out_file = OUT_PATH or f"sp500eu600_value_watchlist_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
    watchlist.write_csv(out_file)
    print(f"Saved watchlist to: {out_file}")

if __name__ == '__main__':
    main()
