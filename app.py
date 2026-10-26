"""Streamlit front door for MFC Stock Forecast.

Run this file with:
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.config import (
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    COMPETITION_END,
    COMPETITION_START,
    PRICE_CACHE_SECONDS,
)
from src.data_store import load_picks, unique_tickers
from src.portfolio import format_return
from src.market_data import fetch_quotes, fetch_year_of_closes
from src.forecast import DEFAULT_SEED, DEFAULT_SIMS, MODEL_VERSION, current_returns, days_left_for, run_forecast
from src.forecast_log import append_forecast, build_log_rows, load_log, market_status
from src.prices import competition_has_ended, competition_has_started, now_et, snapshot_for_tickers
from src.risk import risk_table

# Colors for highlighted students on the chart (colorblind-checked, in this fixed order).
HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
MAX_HIGHLIGHTED = len(HIGHLIGHT_COLORS)

st.set_page_config(page_title="MFC Stock Forecast", layout="wide")
st.title("MFC Stock Forecast")
st.caption("Personal analytics edition · UCI Math Finance Club contest · Oct 20 – Nov 20, 2026")

now = now_et()
started = competition_has_started(now)
ended = competition_has_ended(now)

if not started:
    st.info(
        "The contest starts **Tuesday, October 20, 2026 at 9:30am ET**. "
        "Yahoo can already show current prices. Official start prices lock in at the open."
    )
elif ended:
    st.success("The contest is closed. Returns use the **November 20, 2026** regular-session close.")
else:
    st.warning("Contest is live. Start = Oct 20 open. Current = latest Yahoo last price.")


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def cached_snapshot(tickers: tuple[str, ...]) -> dict:
    return snapshot_for_tickers(list(tickers))


# ---------- 1. Load picks and prices ----------
picks = load_picks()
# Download the students' tickers plus the S&P 500 benchmark in one go.
tickers = sorted(set(unique_tickers(picks)) | {BENCHMARK_TICKER})

if st.button("Refresh prices now"):
    cached_snapshot.clear()

try:
    snapshot = cached_snapshot(tuple(tickers))
except Exception as error:
    st.error(
        "Could not download prices from Yahoo Finance. "
        "Check your Wi-Fi, then click Refresh. Technical detail: "
        f"{error}"
    )
    st.stop()

benchmark = snapshot.get(BENCHMARK_TICKER, {})
benchmark_return = benchmark.get("total_return")

# ---------- 2. Build one row per student ----------
rows = []
for pick in picks:
    data = snapshot.get(pick["ticker"], {})
    total = data.get("total_return")
    rows.append(
        {
            "Student": pick["student"],
            "Company": pick.get("company") or "",
            "Ticker": pick["ticker"],
            "Start (Oct 20 open)": data.get("start_price"),
            "Current / end": data.get("current_price"),
            "Price return": data.get("price_return"),
            "Dividends": data.get("dividend_return"),
            "Total return": total,
            "vs S&P 500": (total - benchmark_return)
            if total is not None and benchmark_return is not None
            else None,
        }
    )

board = pd.DataFrame(rows)
# Best total return first; students with no data yet go to the bottom.
board = board.sort_values("Total return", ascending=False, na_position="last").reset_index(drop=True)
board.insert(0, "Rank", range(1, len(board) + 1))

# ---------- 3. Ticker check ----------
problems = [
    f"**{pick['ticker']}** ({pick['student']}): {snapshot[pick['ticker']]['problem']}"
    for pick in picks
    if snapshot.get(pick["ticker"], {}).get("problem")
]
if problems:
    st.error("Some tickers need attention:\n\n" + "\n".join(f"- {p}" for p in problems))

# ---------- 4. Headline numbers ----------
ranked = board.dropna(subset=["Total return"])
col1, col2, col3 = st.columns(3)
if not ranked.empty:
    leader = ranked.iloc[0]
    col1.metric(
        "🏆 Leader",
        f"{leader['Student']}",
        f"{format_return(leader['Total return'])} · {leader['Ticker']}",
    )
else:
    col1.metric("🏆 Leader", "—")
col2.metric(BENCHMARK_NAME, format_return(benchmark_return))
if benchmark_return is not None and not ranked.empty:
    beating = int((ranked["Total return"] > benchmark_return).sum())
    col3.metric("Beating the market", f"{beating} of {len(ranked)}")
else:
    col3.metric("Beating the market", "—")

leaderboard_tab, chart_tab, quotes_tab, perf_tab, mc_tab = st.tabs(
    ["Leaderboard", "Returns over time", "Quotes & valuation", "1-year performance", "Monte Carlo"]
)

# ---------- 5. Leaderboard table ----------
with leaderboard_tab:

    def highlight_leader(row: pd.Series) -> list[str]:
        style = "background-color: rgba(237, 161, 0, 0.18); font-weight: 600" if row["Rank"] == 1 else ""
        return [style] * len(row)

    def money(value):
        return "—" if pd.isna(value) else f"${value:,.2f}"

    def pct(value):
        return "—" if pd.isna(value) else f"{value:+.2%}"

    # Turn numbers into display text first, so missing values show as "—".
    display = board.copy()
    for column in ["Start (Oct 20 open)", "Current / end"]:
        display[column] = board[column].map(money)
    for column in ["Price return", "Dividends", "Total return", "vs S&P 500"]:
        display[column] = board[column].map(pct)
    styled = display.style.apply(highlight_leader, axis=1)
    st.dataframe(styled, hide_index=True, use_container_width=True, height=36 * (len(board) + 1) + 3)
    st.caption(
        "Ranked by total return = price return + dividends. "
        f"\"vs S&P 500\" is how far ahead (+) or behind (−) of {BENCHMARK_NAME} each pick is. "
        f"Prices refresh about every {PRICE_CACHE_SECONDS} seconds. "
        f"Clock (ET): {now.strftime('%Y-%m-%d %H:%M')}. "
        f"Window: {COMPETITION_START.strftime('%b %d, %Y %H:%M')} → "
        f"{COMPETITION_END.strftime('%b %d, %Y %H:%M')}."
    )

    with st.expander("Check the math by hand"):
        st.markdown(
            """
**Price return** = (current − start) / start

**Dividend return** = dividends paid during the contest / start

**Total return** = price return + dividend return

Example: a stock starts at **$100**, is now **$108**, and paid a **$1.00** dividend
during the contest:

- price return = (108 − 100) / 100 = **+8.00%**
- dividend return = 1 / 100 = **+1.00%**
- total return = **+9.00%**

Only dividends with an ex-dividend date *after* Oct 20 and on or before Nov 20 count.
We don't add the yearly "dividend yield" – that would count a whole year of dividends.

Edit `data/picks.csv` to change students or tickers, then refresh this page.
            """
        )

# ---------- 6. Chart ----------
with chart_tab:
    series_rows = []
    for pick in picks:
        series = snapshot.get(pick["ticker"], {}).get("series")
        if series is None:
            continue
        for day, value in series.items():
            series_rows.append({"Date": day, "Who": pick["student"], "Ticker": pick["ticker"], "Return": value})

    bench_series = benchmark.get("series")
    bench_rows = (
        [{"Date": d, "Who": BENCHMARK_NAME, "Ticker": BENCHMARK_TICKER, "Return": v} for d, v in bench_series.items()]
        if bench_series is not None
        else []
    )

    if not series_rows:
        st.info("The chart appears after the first trading day of the contest.")
    else:
        lines = pd.DataFrame(series_rows)
        top_names = ranked["Student"].head(3).tolist()
        chosen = st.multiselect(
            f"Highlight up to {MAX_HIGHLIGHTED} students (everyone else stays grey)",
            options=board["Student"].tolist(),
            default=top_names,
            max_selections=MAX_HIGHLIGHTED,
        )

        x = alt.X("Date:T", title=None, axis=alt.Axis(format="%b %d", grid=False))
        y = alt.Y("Return:Q", title="Total return", axis=alt.Axis(format="+.0%"))
        tooltip = [
            alt.Tooltip("Who:N", title="Student"),
            alt.Tooltip("Ticker:N"),
            alt.Tooltip("Date:T", format="%b %d"),
            alt.Tooltip("Return:Q", format="+.2%"),
        ]

        zero = alt.Chart(pd.DataFrame({"Return": [0]})).mark_rule(color="#c3c2b7", strokeWidth=1).encode(y="Return:Q")

        everyone = (
            alt.Chart(lines[~lines["Who"].isin(chosen)])
            .mark_line(strokeWidth=1, color="#c3c2b7", opacity=0.8)
            .encode(x=x, y=y, detail="Who:N", tooltip=tooltip)
        )

        # Keep each student's color fixed to their position in the picker.
        highlighted = (
            alt.Chart(lines[lines["Who"].isin(chosen)])
            .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=40, filled=True))
            .encode(
                x=x,
                y=y,
                color=alt.Color(
                    "Who:N",
                    title=None,
                    scale=alt.Scale(domain=chosen, range=HIGHLIGHT_COLORS[: len(chosen)]),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=tooltip,
            )
        )

        layers = [zero, everyone, highlighted]
        if bench_rows:
            market = (
                alt.Chart(pd.DataFrame(bench_rows))
                .mark_line(strokeWidth=2, strokeDash=[6, 4], color="#52514e")
                .encode(x=x, y=y, tooltip=tooltip)
            )
            bench_last = pd.DataFrame(bench_rows).tail(1)
            market_label = (
                alt.Chart(bench_last)
                .mark_text(align="left", dx=6, color="#52514e", fontWeight="bold")
                .encode(x=x, y=y, text=alt.value(BENCHMARK_NAME))
            )
            layers += [market, market_label]

        st.altair_chart(alt.layer(*layers).properties(height=460), use_container_width=True)
        st.caption(
            "Each line is a pick's total return at each day's close. "
            f"The dashed line is the market ({BENCHMARK_NAME}). Hover any line for details."
        )


# ---------- V2 helpers ----------
student_of = {pick["ticker"]: pick["student"] for pick in picks}
pick_tickers = [pick["ticker"] for pick in picks]


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def cached_quotes(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_quotes(list(tickers))


@st.cache_data(ttl=3600)  # a year of history barely changes within an hour
def cached_year(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_year_of_closes(list(tickers))


def fmt_money(value):
    return "—" if pd.isna(value) else f"${value:,.2f}"


def fmt_pct(value, signed=True):
    if pd.isna(value):
        return "—"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def fmt_number(value, decimals=1):
    return "—" if pd.isna(value) else f"{value:,.{decimals}f}"


def fmt_big_money(value):
    if pd.isna(value):
        return "—"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.1f}{suffix}"
    return f"${value:,.0f}"


# ---------- 7. Quotes & valuation ----------
with quotes_tab:
    st.subheader("Quotes & valuation")
    try:
        quotes = cached_quotes(tuple(pick_tickers))
    except Exception as error:
        st.error(f"Could not load quotes from Yahoo Finance: {error}")
        quotes = pd.DataFrame()

    if not quotes.empty:
        quotes.insert(0, "Student", quotes["Ticker"].map(student_of))
        columns = [
            "Student", "Ticker", "Price", "Bid", "Ask", "Spread ($)", "Spread (%)",
            "P/E (trailing)", "P/E (forward)", "Market cap", "Beta", "52-wk low", "52-wk high", "Sector",
        ]
        table = quotes[columns]
        # Turn every number into display text ("—" when Yahoo has nothing).
        formatters = {
            "Price": fmt_money, "Bid": fmt_money, "Ask": fmt_money,
            "52-wk low": fmt_money, "52-wk high": fmt_money,
            "Spread ($)": lambda v: "—" if pd.isna(v) else f"${v:,.3f}",
            "Spread (%)": lambda v: fmt_pct(v, signed=False),
            "P/E (trailing)": fmt_number, "P/E (forward)": fmt_number,
            "Beta": lambda v: fmt_number(v, 2),
            "Market cap": fmt_big_money,
            "Sector": lambda v: "—" if v is None or pd.isna(v) else v,
        }
        styled_quotes = table.copy()
        for column, formatter in formatters.items():
            styled_quotes[column] = table[column].map(formatter)
        st.dataframe(styled_quotes, hide_index=True, use_container_width=True, height=36 * (len(table) + 1) + 3)

        if table[["Bid", "Ask"]].isna().all().all():
            st.info(
                "Bid and ask are blank because the market is closed. Yahoo only shows live quotes "
                "Monday–Friday, 9:30am–4pm ET."
            )

        with st.expander("What do these mean?"):
            st.markdown(
                """
- **Bid** – the highest price someone is currently willing to *pay*. **Ask** – the lowest price someone is willing to *sell* at.
- **Spread** = ask − bid. **Spread %** = spread ÷ midpoint, where midpoint = (bid + ask) ÷ 2.
  A small spread means the stock is easy to trade (liquid). Big, popular stocks often have spreads under 0.05%.
- **P/E (trailing)** = price ÷ the last 12 months of earnings per share. **P/E (forward)** uses analysts' estimate
  of next year's earnings. Blank means the company isn't profitable (P/E isn't meaningful with negative earnings).
- **Beta** – how much the stock tends to move when the market moves 1%. Beta 2 ≈ twice as jumpy as the market.
- **Market cap** = share price × number of shares: the company's total stock-market value.
                """
            )

# ---------- 8. One-year performance ----------
with perf_tab:
    st.subheader("Last 12 months")
    try:
        year = cached_year(tuple(sorted(set(pick_tickers) | {BENCHMARK_TICKER})))
    except Exception as error:
        st.error(f"Could not load price history from Yahoo Finance: {error}")
        year = pd.DataFrame()

    if year.empty:
        st.info("No price history available yet.")
    else:
        available = [t for t in pick_tickers if t in year.columns]
        label_of = {t: f"{student_of[t]} ({t})" for t in available}
        chosen_tickers = st.multiselect(
            f"Compare up to {MAX_HIGHLIGHTED} picks against the market",
            options=available,
            default=available[:3],
            format_func=lambda t: label_of[t],
            max_selections=MAX_HIGHLIGHTED,
        )

        # Growth of $100: divide every price by the first price, times 100.
        growth = year / year.apply(lambda col: col.dropna().iloc[0]) * 100
        growth.index.name = "Date"
        long = growth.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Value").dropna()

        x = alt.X("Date:T", title=None, axis=alt.Axis(format="%b %Y", grid=False, tickCount="month"))
        y = alt.Y(
            "Value:Q",
            title="Value of $100 invested a year ago",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(zero=False),
        )
        tip = [alt.Tooltip("Ticker:N"), alt.Tooltip("Date:T", format="%b %d, %Y"), alt.Tooltip("Value:Q", format="$,.2f")]

        base_line = alt.Chart(pd.DataFrame({"Value": [100]})).mark_rule(color="#c3c2b7").encode(y="Value:Q")
        picked = (
            alt.Chart(long[long["Ticker"].isin(chosen_tickers)])
            .mark_line(strokeWidth=2)
            .encode(
                x=x,
                y=y,
                color=alt.Color(
                    "Ticker:N",
                    title=None,
                    scale=alt.Scale(domain=chosen_tickers, range=HIGHLIGHT_COLORS[: len(chosen_tickers)]),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=tip,
            )
        )
        market_line = (
            alt.Chart(long[long["Ticker"] == BENCHMARK_TICKER])
            .mark_line(strokeWidth=2, strokeDash=[6, 4], color="#52514e")
            .encode(x=x, y=y, tooltip=tip)
        )
        st.altair_chart(alt.layer(base_line, market_line, picked).properties(height=420), use_container_width=True)
        st.caption(f"Prices include dividends and splits. Dashed line = {BENCHMARK_NAME}.")

        st.subheader("Risk check")
        risks = risk_table(year)
        risks.insert(0, "Student", risks["Ticker"].map(lambda t: student_of.get(t, BENCHMARK_NAME)))
        risks = risks.sort_values("Volatility (yearly)", ascending=False)
        st.dataframe(
            risks.style.format(fmt_pct, subset=["1-year return", "Max drawdown", "Best day", "Worst day"]).format(
                lambda v: fmt_pct(v, signed=False), subset=["Volatility (yearly)"]
            ),
            hide_index=True,
            use_container_width=True,
            height=36 * (len(risks) + 1) + 3,
        )
        with st.expander("What do these mean?"):
            st.markdown(
                """
- **Volatility (yearly)** – how much the stock typically swings in a year
  (standard deviation of daily returns × √252). Higher = riskier, but also more chance of a big win in a short contest.
- **Max drawdown** – the worst drop from a previous high during the year.
- **Best / worst day** – the biggest one-day gain and loss.
                """
            )

# ---------- 9. Monte Carlo ----------
@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def cached_forecast(current: pd.Series, year_closes: pd.DataFrame, days_left: int, n_sims: int, seed: int) -> pd.DataFrame:
    return run_forecast(current, year_closes, days_left, n_sims=n_sims, seed=seed)


with mc_tab:
    st.subheader("Who's most likely to win?")
    st.write(
        "We replay the rest of the contest thousands of times using real daily moves from the past year, "
        "then count how often each student finishes first."
    )

    try:
        year_mc = cached_year(tuple(sorted(set(pick_tickers) | {BENCHMARK_TICKER})))
    except Exception:
        year_mc = pd.DataFrame()

    if year_mc.empty:
        st.info("Needs a year of price history from Yahoo Finance.")
    else:
        days_left = days_left_for(now)
        current = current_returns(picks, snapshot, year_mc)

        n_sims = st.select_slider(
            "Number of simulations", options=[1_000, 5_000, 10_000, 50_000], value=DEFAULT_SIMS
        )
        forecast = cached_forecast(current, year_mc, days_left, n_sims, DEFAULT_SEED)

        c1, c2 = st.columns(2)
        c1.metric("Trading days left", days_left)
        favorite = forecast.index[0]
        c2.metric(
            "Favorite to win",
            student_of[favorite],
            f"{forecast['win_prob'].iloc[0]:.0%} chance · {favorite}",
            delta_color="off",
        )

        results = pd.DataFrame(
            {
                "Student": [student_of[t] for t in forecast.index],
                "Ticker": forecast.index,
                "Win chance": forecast["win_prob"].values,
                "Return so far": forecast["return_so_far"].values,
                "Bad case (5th pct)": forecast["p05"].values,
                "Typical (median)": forecast["median"].values,
                "Good case (95th pct)": forecast["p95"].values,
            }
        )

        bars = (
            alt.Chart(results)
            .mark_bar(color="#2a78d6", cornerRadiusEnd=4, height=16)
            .encode(
                y=alt.Y("Student:N", sort="-x", title=None),
                x=alt.X("Win chance:Q", title="Chance of finishing first", axis=alt.Axis(format=".0%", tickCount=5)),
                tooltip=[
                    alt.Tooltip("Student:N"),
                    alt.Tooltip("Ticker:N"),
                    alt.Tooltip("Win chance:Q", format=".1%"),
                    alt.Tooltip("Return so far:Q", format="+.2%"),
                ],
            )
        )
        labels = bars.mark_text(align="left", dx=4, color="#52514e").encode(
            text=alt.Text("Win chance:Q", format=".1%")
        )
        st.altair_chart((bars + labels).properties(height=26 * len(results) + 40), use_container_width=True)

        st.subheader("Range of likely finishes")
        display_results = results.copy()
        for column in ["Return so far", "Bad case (5th pct)", "Typical (median)", "Good case (95th pct)"]:
            display_results[column] = results[column].map(fmt_pct)
        display_results["Win chance"] = results["Win chance"].map(lambda v: f"{v:.1%}")
        st.dataframe(display_results, hide_index=True, use_container_width=True, height=36 * (len(results) + 1) + 3)

        with st.expander("How the simulation works (and its limits)"):
            st.markdown(
                f"""
1. Start from each pick's return so far.
2. For each of the **{days_left}** trading days left, pick a random real trading day from the past year
   and apply every stock's move from that day. Using the *same* day for everyone keeps stocks that move
   together (for example chip makers) moving together.
3. Repeat **{n_sims:,}** times and count who finishes first.

**Limits:** the past year may not look like the next few weeks; earnings announcements and news can move
a stock far more than a typical day; and future dividends aren't included. Treat these as rough odds, not predictions.
                """
            )

        # ---------- 10. Forecast log ----------
        st.divider()
        st.subheader("Forecast history")
        st.write(
            "Every trading day, the official forecast "
            f"({DEFAULT_SIMS:,} simulations, model `{MODEL_VERSION}`) is saved to `data/forecast_log.csv`. "
            "On Nov 20 this shows whether the model saw the winner coming, and how early."
        )

        if st.button("Log today's forecast"):
            official = cached_forecast(current, year_mc, days_left, DEFAULT_SIMS, DEFAULT_SEED)
            rows = build_log_rows(
                official,
                student_of=student_of,
                now=now,
                days_left=days_left,
                model=MODEL_VERSION,
                n_sims=DEFAULT_SIMS,
                seed=DEFAULT_SEED,
                spy_return=benchmark_return,
            )
            saved, message = append_forecast(rows)
            (st.success if saved else st.info)(message)

        if market_status(now) == "intraday":
            st.caption("The market is open right now. The official daily forecast is logged after the 4pm ET close.")

        log = load_log()
        if log.empty:
            st.info(
                "No forecasts logged yet. Click **Log today's forecast** after the market closes, "
                "or let the daily GitHub Action do it."
            )
        else:
            days_logged = log["forecast_date"].nunique()
            latest_day = log["forecast_date"].max()
            st.caption(f"{days_logged} day(s) logged · latest: {latest_day}")

            # Follow the students the model currently likes most.
            latest = log[log["forecast_date"] == latest_day].sort_values("win_prob", ascending=False)
            top = latest["student"].head(MAX_HIGHLIGHTED).tolist()
            follow = st.multiselect(
                f"Students to follow (up to {MAX_HIGHLIGHTED})",
                options=sorted(log["student"].dropna().unique()),
                default=top,
                max_selections=MAX_HIGHLIGHTED,
            )
            history = log[log["student"].isin(follow)].copy()
            # One evenly spaced point per trading day (weekends are skipped, not drawn as gaps).
            history["day"] = pd.to_datetime(history["forecast_date"]).dt.strftime("%b %d")
            day_order = sorted(history["forecast_date"].unique())
            day_labels = [pd.Timestamp(d).strftime("%b %d") for d in day_order]

            if days_logged >= 2:
                odds_chart = (
                    alt.Chart(history)
                    .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True))
                    .encode(
                        x=alt.X("day:O", title=None, sort=day_labels, axis=alt.Axis(labelAngle=0, grid=False)),
                        y=alt.Y("win_prob:Q", title="Chance of winning", axis=alt.Axis(format=".0%")),
                        color=alt.Color(
                            "student:N",
                            title=None,
                            scale=alt.Scale(domain=follow, range=HIGHLIGHT_COLORS[: len(follow)]),
                            legend=alt.Legend(orient="top"),
                        ),
                        tooltip=[
                            alt.Tooltip("student:N", title="Student"),
                            alt.Tooltip("ticker:N"),
                            alt.Tooltip("day:N", title="Date"),
                            alt.Tooltip("win_prob:Q", title="Win chance", format=".1%"),
                            alt.Tooltip("return_so_far:Q", title="Return so far", format="+.2%"),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(odds_chart, use_container_width=True)
            else:
                st.info("The chart of odds over time appears once there are at least 2 days in the log.")

            st.download_button(
                "Download the full log (CSV)",
                data=log.to_csv(index=False).encode("utf-8"),
                file_name="forecast_log.csv",
                mime="text/csv",
            )
