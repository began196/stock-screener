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

# For quick debugging: set to e.g. 30, then None when ready
MAX_TICKERS = None

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

def to_billions(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x / 1e9

# Fetch SnP500 tickers
def fetch_sp500_tickers() -> pd.DataFrame:
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

    return sp500

# Fetch STOXX Europe 600 tickers
def read_ishares_holdings_csv(text: str) -> pd.DataFrame:
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

    # Clean column names (sometimes extra whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df

def fetch_stoxx600_from_ishares() -> pd.DataFrame:
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
        "Country": df["Location"].astype(str).str.strip(),
        "Region": "Europe"
    })

    out = out[out["Ticker"].notna() & (out["Ticker"] != "")]
    out = out.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return out

# Discounted cash flow and peer multiple valuation functions
def dcf_lite_equity_value_per_share(
    fcf: float,
    shares_out: float,
    net_debt: float,
    params: DCFParams,
    growth: float,
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
    peer_ev_to_ebitda_median: float,
    ebitda: float,
    shares_out: float,
    net_debt: float,
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
def build_universe_snapshot(tickers: List[str]) -> pd.DataFrame:
    rows = []

    for i, t in enumerate(tickers, 1):
        try:
            tk = yf.Ticker(t)
            info = tk.get_info()

            price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
            market_cap = safe_get(info, "marketCap")
            enterprise_value = safe_get(info, "enterpriseValue")
            ev_to_ebitda = safe_get(info, "enterpriseToEbitda")
            sector = safe_get(info, "sector")

            q = compute_quality_metrics(info, tk)

            rows.append(
                {
                    "ticker": t,
                    "sector": sector,
                    "price": price,
                    "market_cap": market_cap,
                    "enterprise_value": enterprise_value,
                    "ev_to_ebitda": ev_to_ebitda,
                    **q,
                }
            )
        except Exception as e:
            rows.append({"ticker": t, "error": str(e)})

        if i % 50 == 0:
            print(f"Fetched {i}/{len(tickers)} tickers...")

    return pd.DataFrame(rows)

# Quality screening of stocks
def apply_quality_screen(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["qc_fcf_pos"] = d["fcf"].fillna(-1) > 0
    d["qc_ebitda_pos"] = d["ebitda"].fillna(-1) > 0

    nde = d["net_debt_to_ebitda"]
    d["qc_leverage_ok"] = nde.isna() | (nde <= 3.0)

    ic = d["interest_coverage"]
    d["qc_intcov_ok"] = ic.isna() | (ic >= 3.0)

    roic = d["roic_proxy"]
    d["qc_roic_ok"] = roic.isna() | (roic >= 0.08)

    d["quality_pass"] = (
        d["qc_fcf_pos"] &
        d["qc_ebitda_pos"] &
        d["qc_leverage_ok"] &
        d["qc_intcov_ok"] &
        d["qc_roic_ok"]
    )

    return d

# Compute intrinsic value of the companies and generate trading singals
def compute_intrinsic_and_signals(
    df: pd.DataFrame,
    dcf_params: DCFParams,
    sig_params: SignalParams,
) -> pd.DataFrame:
    d = df.copy()

    # Peer EV/EBITDA median across the universe
    peer_pool = d["ev_to_ebitda"].replace([np.inf, -np.inf], np.nan)
    peer_pool = peer_pool[(peer_pool > 0) & (peer_pool < 80)]
    peer_median = float(peer_pool.median()) if len(peer_pool) else np.nan

    d["peer_ev_to_ebitda_median"] = peer_median

    dcf_bear, dcf_base, dcf_bull, peer_fv = [], [], [], []

    for _, r in d.iterrows():
        fcf = r.get("fcf")
        shares = r.get("shares_out")
        net_debt = r.get("net_debt")
        ebitda = r.get("ebitda")

        # DCF
        if any(pd.isna(x) for x in [fcf, shares, net_debt]):
            dcf_bear.append(np.nan); dcf_base.append(np.nan); dcf_bull.append(np.nan)
        else:
            dcf_bear.append(dcf_lite_equity_value_per_share(float(fcf), float(shares), float(net_debt), dcf_params, dcf_params.growth_bear))
            dcf_base.append(dcf_lite_equity_value_per_share(float(fcf), float(shares), float(net_debt), dcf_params, dcf_params.growth_base))
            dcf_bull.append(dcf_lite_equity_value_per_share(float(fcf), float(shares), float(net_debt), dcf_params, dcf_params.growth_bull))

        # Peer multiple
        if not pd.isna(peer_median) and not any(pd.isna(x) for x in [ebitda, shares, net_debt]):
            peer_fv.append(peer_multiple_fair_value_per_share(peer_median, float(ebitda), float(shares), float(net_debt)))
        else:
            peer_fv.append(np.nan)

    d["dcf_bear"] = dcf_bear
    d["dcf_base"] = dcf_base
    d["dcf_bull"] = dcf_bull
    d["peer_fair_value"] = peer_fv

    # Blend intrinsic estimate
    intrinsic = []
    for _, r in d.iterrows():
        a, b = r.get("dcf_base"), r.get("peer_fair_value")
        if pd.notna(a) and pd.notna(b):
            intrinsic.append(0.6 * float(a) + 0.4 * float(b))
        elif pd.notna(a):
            intrinsic.append(float(a))
        elif pd.notna(b):
            intrinsic.append(float(b))
        else:
            intrinsic.append(np.nan)

    d["intrinsic_est"] = intrinsic

    mos = sig_params.margin_of_safety
    buf = sig_params.sell_buffer

    d["buy_threshold"] = d["intrinsic_est"] * (1.0 - mos)
    d["sell_threshold"] = d["intrinsic_est"] * (1.0 + buf)

    def signal_row(r) -> str:
        price = r.get("price")
        intrinsic_est = r.get("intrinsic_est")
        if pd.isna(price) or pd.isna(intrinsic_est):
            return "NO_DATA"
        if not bool(r.get("quality_pass", False)):
            if price <= r.get("buy_threshold", -np.inf):
                return "CHEAP_BUT_FAILS_QUALITY"
            return "FAILS_QUALITY"
        if price <= r.get("buy_threshold"):
            return "BUY"
        if price >= r.get("sell_threshold"):
            return "SELL/TRIM"
        return "HOLD"

    d["signal"] = d.apply(signal_row, axis=1)
    d["upside_to_intrinsic"] = (d["intrinsic_est"] / d["price"]) - 1.0

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

    # Create universe dataframe
    sp500_universe = sp500.copy()
    sp500_universe["Country"] = "United States"
    sp500_universe["Region"] = "US"

    us = sp500_universe[["Ticker", "Security", "Country", "Region", "GICS Sector", "GICS Sub-Industry"]].copy()

    # Europe holdings do not have GICS sub-industry
    eu = europe.copy()
    eu["GICS Sub-Industry"] = None

    universe = pd.concat([us, eu], ignore_index=True)

    # Get ticker symbols
    snp500_tickers = sp500["Ticker"].tolist()
    europe_tickers = europe["Ticker"].tolist()
    if MAX_TICKERS is not None:
        snp500_tickers = snp500_tickers[:MAX_TICKERS]
        europe_tickers = europe_tickers[:MAX_TICKERS]

    tickers = snp500_tickers + europe_tickers

    print(f"Loaded {len(tickers)} S&P 500 + STOXX Europe 600 tickers.", file=sys.stderr)

    df = build_universe_snapshot(tickers)

    # Merge df with company details
    df = df.merge(
        universe[["Ticker", "Security", "GICS Sector", "GICS Sub-Industry", "Country", "Region"]],
        left_on="ticker",
        right_on="Ticker",
        how="left",
    ).drop(columns=["Ticker"])

    df = apply_quality_screen(df)

    # Calculate intrinsic value and generate signals based on dcf and signal hyperparameters
    df = compute_intrinsic_and_signals(df, dcf_params, sig_params)
    df[["ticker","price","intrinsic_est","upside_to_intrinsic","signal","quality_pass"]].sort_values("upside_to_intrinsic", ascending=False).head(20)

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

    watchlist = df[watchlist_cols].copy()

    # add readability columns
    watchlist["fcf_bil"] = watchlist["fcf"].apply(to_billions)
    watchlist["ebitda_bil"] = watchlist["ebitda"].apply(to_billions)

    out_file = OUT_PATH or f"sp500eu600_value_watchlist_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}.csv"
    watchlist.to_csv(out_file, index=False)
    print(f"Saved watchlist to: {out_file}")

if __name__ == '__main__':
    main()
