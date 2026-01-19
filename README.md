# stock-screener
Value Investing Screener for S&amp;P 500 and STOXX Europe 600

# Value Stock Screening Methodology

This stock screener identifies **potential value investment opportunities** by combining **fundamental valuation**, **financial quality checks**, and **rule-based buy/sell discipline**. 

## Quick-Start Guide

Run the script directly from your terminal:

```bash
python stockscreener.py
```

You can provide non-default hyperparmeters using the arguments below:

- out: None (default). Output csv file name
- max: None (default). Select the top `max` stocks for S&amp;P 500 and STOXX Europe 600. None selects all
- discount: 0.10 (default). Discount rate. Higher values make valuations more conservative.
- terminal_growth: 0.02 (default). Terminal growth rate. Lower values assume slower long-term growth.
- mos: 0.30 (default). Margin of safety. Higher values demand deeper undervaluation before a stock is labelled BUY.
- sell_buffer: 0.15 (default). Sell Buffer. 

For example to change the terminal_growth and margin of safety parameters:

```bash
python stockscreener.py --terminal_growth 0.03 --mos 0.20
```

---

## Step 1: Business Quality Filter

Before valuing a company, the screener checks whether the underlying business is financially sound. 

A company must demonstrate:
- **Positive Free Cash Flow** — the business generates real cash
- **Positive EBITDA** — the core business is operationally viable
- **Manageable leverage** — Net Debt / EBITDA ≤ 3 (when available)
- **Adequate interest coverage** — ability to service debt
- **Acceptable return on invested capital (ROIC)** where data is available

Companies that fail these checks are flagged and are never labelled as **BUY**, even if they appear cheap.

---

## Step 2: Intrinsic Value Estimation

For companies that pass the quality screen, intrinsic value is estimated using two valuation methods.

### Discounted Cash Flow (DCF)

- Projects future free cash flows over a fixed horizon
- Discounts them back to today using a required return
- Applies a conservative terminal growth rate
- Uses multiple scenarios (bear / base / bull) to avoid false precision

This estimates what the business is worth based on its ability to generate cash for owners.

---

### Peer Multiple Valuation (EV / EBITDA)

- Compares each company to similar businesses in the same universe
- Uses the median peer multiple to reduce outlier distortion
- Converts enterprise value to equity value using net debt

This reflects how the market typically prices comparable businesses.

The screener blends both approaches to produce a **robust intrinsic value estimate**.

---

## Step 3: Margin of Safety & Signals

The final decision is rule-based:

- **BUY**  
  When the market price is significantly below intrinsic value  
  *(by at least the margin of safety)* and the business passes quality checks

- **HOLD**  
  When price is close to fair value or upside is limited

- **SELL / TRIM**  
  When price exceeds intrinsic value by a defined buffer

This enforces discipline and removes emotional decision-making.


