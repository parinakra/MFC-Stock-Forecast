# MFC Stock Forecast

A live stock-competition dashboard with risk analytics and a Monte Carlo win-probability model, built in Python with Streamlit, pandas, NumPy, and Yahoo Finance data.

Version 1 is the leaderboard I run for the **UCI Math Finance Club** 5-week stock contest (Oct 20 – Nov 20, 2026): each member nominates one stock, and the highest total return wins. MFC Stock Forecast is my personal analytics project built on top of it: it forecasts each pick's chance of winning, logs those forecasts every trading day, and evaluates them when the contest ends.

## Features

Headline numbers (leader, favorite to win, S&P 500, picks beating the market) above six tabs:

| Tab | What it shows |
| --- | --- |
| **Leaderboard** | Start price, current price, price return, dividends, total return, vs S&P 500 |
| **Returns over time** | Daily total return for every pick, top picks highlighted, S&P 500 dashed |
| **Quotes & valuation** | Price, bid, ask, bid–ask spread, trailing and forward P/E, market cap, beta, 52-week range |
| **1-year performance** | Growth of $100 vs the S&P 500; volatility, max drawdown, best and worst day |
| **Monte Carlo** | Win chance for every pick, likely finish range, win chance over time (from the daily log) |
| **Model comparison** | Each model's favorite, and every pick's win chance under all five models |

## Methods

**Total return** = (current price − start price + dividends paid) / start price. Start = Oct 20 regular-session open. Only dividends with an ex-date after Oct 20 and on or before Nov 20 count.

**Bid–ask spread** = ask − bid; spread % = spread / midpoint. Shown only while the market is open (Yahoo reports 0 otherwise).

**Volatility** = standard deviation of daily returns × √252. **Max drawdown** = worst fall from a running high.

**Monte Carlo (bootstrap):** for each remaining trading day, draw a random real trading day from the past year and apply *every* stock's return from that day. Resampling whole days keeps correlations between stocks (e.g. semiconductors moving together) and real-world fat tails, without assuming returns follow a bell curve. Repeat 10,000 times and count how often each pick finishes first.

**Five models, side by side** (`src/models.py`). Each makes a different assumption about how returns behave:

| Model | Assumption |
| --- | --- |
| Historical days (bootstrap, official) | Replay random real trading days; keeps real correlations and real extreme days |
| Bell curve (GBM) | Multivariate normal log-returns with the past year's means, volatilities and correlations |
| Fat tails (Student-t, 4 d.o.f.) | Same means, volatilities and correlations as the bell curve, but more extreme days |
| Historical weeks (block bootstrap) | Replay real 5-day stretches, preserving volatility clustering |
| Market + own noise (factor model) | Each stock = alpha + beta × S&P 500 + its own independent noise |

If the models agree, the forecast is robust; where they disagree, the answer depends on the assumption.

**Forecast log:** every weekday after the 4pm ET close, a GitHub Action runs `log_forecast.py` and appends every model's forecast (10,000 simulations each, fixed seed, model version recorded) to `data/forecast_log.csv`. Rules: one forecast per model per trading day, past forecasts are never edited, and everything needed to reproduce a forecast is stored with it. When the contest ends, the log is used to evaluate the model: did each model favor the eventual winner, how early, and which model's probabilities were best calibrated?

Limits: the past year may not resemble the next few weeks, earnings and news can move a stock far more than a typical day, and future dividends aren't modeled.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit app |
| `data/picks.csv` | Members and tickers |
| `src/config.py` | Contest dates, benchmark ticker, file paths |
| `src/data_store.py` | Reads `picks.csv` |
| `src/prices.py` | Contest prices and dividends from Yahoo Finance |
| `src/portfolio.py` | Price, dividend, and total return formulas |
| `src/market_data.py` | Quotes, valuation fields, one year of history |
| `src/risk.py` | Volatility, drawdown, period return |
| `src/monte_carlo.py` | Simulation engine and win probabilities |
| `src/models.py` | The five forecasting models |
| `src/forecast.py` | Builds the forecast used by both the app and the logger |
| `src/forecast_log.py` | Reads and appends the forecast log (one entry per trading day) |
| `log_forecast.py` | Script that saves today's forecast |
| `data/forecast_log.csv` | The forecast history (created by the first logged forecast) |
| `.github/workflows/log_forecast.yml` | Runs the logger automatically every weekday after the close |
| `tests/` | Calculator-checkable tests for all the math (no internet needed) |

## Run it (Mac)

```bash
cd "/Users/parinakra/Downloads/MFC Stock Monitor V2"
python3 -m venv .venv            # or /opt/anaconda3/bin/python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens `http://localhost:8501`. Stop with **Control + C**.

Run the tests:

```bash
python tests/test_portfolio.py
python tests/test_analytics.py
python tests/test_forecast_log.py
python tests/test_models.py
```

Log today's forecast by hand (after 4pm ET / 1pm PT):

```bash
python log_forecast.py
```

## Ideas for next versions

- Backtest: how would each pick have done in past 5-week windows?
- Sharpe ratio and correlation heatmap across picks
- Fan chart of simulated paths for one stock
- Deploy publicly on Streamlit Community Cloud
