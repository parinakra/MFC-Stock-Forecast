"""MFC Stock Forecast: one-page dashboard.

Run with:
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.config import BENCHMARK_NAME, BENCHMARK_TICKER, PRICE_CACHE_SECONDS
from src.data_store import load_picks
from src.forecast import DEFAULT_SEED, DEFAULT_SIMS, MODEL_VERSION, compare_models, current_returns, days_left_for, run_forecast
from src.forecast_log import load_log
from src.market_data import fetch_quotes, fetch_year_of_closes
from src.models import MODELS
from src.prices import now_et, snapshot_for_tickers
from src.risk import risk_table

# Colorblind-checked series colors, always used in this order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SHAPES = ["circle", "square", "triangle-up", "diamond", "cross"]
MARKET_GREY = "#52514e"
TOP_N = 5

st.set_page_config(page_title="MFC Stock Forecast", layout="wide")


# ---------- Data (cached so the page stays fast) ----------
@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def get_snapshot(tickers: tuple[str, ...]) -> dict:
    return snapshot_for_tickers(list(tickers))


@st.cache_data(ttl=3600)
def get_year(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_year_of_closes(list(tickers))


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def get_forecast(current: pd.Series, year: pd.DataFrame, days_left: int) -> pd.DataFrame:
    return run_forecast(current, year, days_left, n_sims=DEFAULT_SIMS, seed=DEFAULT_SEED)


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def get_model_comparison(current: pd.Series, year: pd.DataFrame, days_left: int) -> pd.DataFrame:
    return compare_models(current, year, days_left, n_sims=DEFAULT_SIMS, seed=DEFAULT_SEED)


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def get_quotes(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_quotes(list(tickers))


picks = load_picks()
student_of = {p["ticker"]: p["student"] for p in picks}
company_of = {p["ticker"]: p.get("company", "") for p in picks}
pick_tickers = [p["ticker"] for p in picks]
all_tickers = tuple(sorted(set(pick_tickers) | {BENCHMARK_TICKER}))
now = now_et()

try:
    snapshot = get_snapshot(all_tickers)
    year = get_year(all_tickers)
except Exception as error:
    st.error(f"Couldn't load prices from Yahoo Finance ({error}). Check your connection and refresh.")
    st.stop()

spy = snapshot.get(BENCHMARK_TICKER, {}).get("total_return")
days_left = days_left_for(now)
current = current_returns(picks, snapshot, year)
forecast = get_forecast(current, year, days_left) if not year.empty else pd.DataFrame()

# ---------- Header ----------
title_col, button_col = st.columns([5, 1])
title_col.title("MFC Stock Forecast")
title_col.caption(f"As of {now:%a %b %d, %I:%M %p} ET · {days_left} trading days left · contest ends Nov 20")
if button_col.button("Refresh", use_container_width=True):
    get_snapshot.clear()
    get_forecast.clear()
    get_model_comparison.clear()
    st.rerun()

problems = [f"{t} ({student_of[t]})" for t in pick_tickers if snapshot.get(t, {}).get("problem")]
if problems:
    st.error("No price data: " + ", ".join(problems))

# ---------- Main table: leaderboard + win odds ----------
rows = []
for t in pick_tickers:
    data = snapshot.get(t, {})
    total = data.get("total_return")
    rows.append(
        {
            "Student": student_of[t],
            "Ticker": t,
            "Company": company_of[t],
            "Return": total * 100 if total is not None else None,
            "vs S&P 500": (total - spy) * 100 if total is not None and spy is not None else None,
            "Win chance": forecast["win_prob"].get(t, 0.0) * 100 if not forecast.empty else None,
        }
    )
board = pd.DataFrame(rows).sort_values("Return", ascending=False, na_position="last").reset_index(drop=True)
board.insert(0, "Rank", range(1, len(board) + 1))

ranked = board.dropna(subset=["Return"])
k1, k2, k3, k4 = st.columns(4)
if not ranked.empty:
    leader = ranked.iloc[0]
    k1.metric("Leader", f"{leader['Student']} · {leader['Ticker']}", f"{leader['Return']:+.2f}%")
if not forecast.empty:
    fav = forecast.index[0]
    k2.metric("Favorite to win", f"{student_of[fav]} · {fav}", f"{forecast['win_prob'].iloc[0]:.1%} chance", delta_color="off")
k3.metric(BENCHMARK_NAME, f"{spy:+.2%}" if spy is not None else "—")
if spy is not None and not ranked.empty:
    k4.metric("Beating the S&P", f"{int((ranked['vs S&P 500'] > 0).sum())} of {len(ranked)}")

max_odds = float(board["Win chance"].max() or 1)
st.dataframe(
    board,
    hide_index=True,
    use_container_width=True,
    height=35 * (len(board) + 1) + 3,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Return": st.column_config.NumberColumn(format="%+.2f%%"),
        "vs S&P 500": st.column_config.NumberColumn(format="%+.2f%%"),
        "Win chance": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=max_odds),
    },
)

# ---------- Everything else, collapsed ----------
log = load_log()
official_log = log[log["model"] == MODEL_VERSION] if not log.empty else log

with st.expander("Win chance over time"):
    if official_log["forecast_date"].nunique() < 2 if not official_log.empty else True:
        st.write("Needs at least 2 logged days.")
    else:
        latest_day = official_log["forecast_date"].max()
        default = (
            official_log[official_log["forecast_date"] == latest_day]
            .sort_values("win_prob", ascending=False)["student"].head(TOP_N).tolist()
        )
        follow = st.multiselect("Students", sorted(official_log["student"].unique()), default=default, max_selections=TOP_N)
        hist = official_log[official_log["student"].isin(follow)].copy()
        hist["Day"] = pd.to_datetime(hist["forecast_date"]).dt.strftime("%b %d")
        day_order = [pd.Timestamp(d).strftime("%b %d") for d in sorted(hist["forecast_date"].unique())]
        chart = (
            alt.Chart(hist)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True))
            .encode(
                x=alt.X("Day:O", sort=day_order, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("win_prob:Q", title="Win chance", axis=alt.Axis(format=".0%")),
                color=alt.Color("student:N", title=None, scale=alt.Scale(domain=follow, range=COLORS[: len(follow)]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("student:N", title="Student"), "ticker:N", "Day:N",
                         alt.Tooltip("win_prob:Q", title="Win chance", format=".1%")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

with st.expander("Model comparison"):
    if year.empty:
        st.write("No price history.")
    else:
        odds = get_model_comparison(current, year, days_left)
        names = [MODELS[k][0] for k in MODELS]
        table = odds.copy() * 100
        table.columns = names
        table["Spread"] = table.max(axis=1) - table.min(axis=1)
        table.insert(0, "Student", [student_of[t] for t in table.index])
        table.insert(1, "Ticker", table.index)
        table = table.sort_values(names[0], ascending=False)

        top = table.head(10)
        dots = top.melt(id_vars=["Student", "Ticker"], value_vars=names, var_name="Model", value_name="Win chance")
        dots["Pick"] = dots["Student"] + " · " + dots["Ticker"]
        order = (top["Student"] + " · " + top["Ticker"]).tolist()
        bars = top.assign(Pick=order, low=top[names].min(axis=1), high=top[names].max(axis=1))
        chart = alt.layer(
            alt.Chart(bars).mark_rule(color="#c3c2b7", strokeWidth=6, strokeCap="round", opacity=0.6).encode(
                y=alt.Y("Pick:N", sort=order, title=None), x=alt.X("low:Q", title="Win chance (%)", axis=alt.Axis(tickCount=6)), x2="high:Q"
            ),
            alt.Chart(dots).mark_point(size=90, filled=True, stroke="white", strokeWidth=1).encode(
                y=alt.Y("Pick:N", sort=order),
                x="Win chance:Q",
                color=alt.Color("Model:N", scale=alt.Scale(domain=names, range=COLORS),
                                legend=alt.Legend(orient="top", title=None, symbolSize=120)),
                shape=alt.Shape("Model:N", scale=alt.Scale(domain=names, range=SHAPES),
                                legend=alt.Legend(orient="top", title=None, symbolSize=120)),
                tooltip=["Pick:N", "Model:N", alt.Tooltip("Win chance:Q", format=".1f")],
            ),
        ).properties(height=34 * len(top) + 60)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in names}
            | {"Spread": st.column_config.NumberColumn(format="%.1f pts")},
        )

with st.expander("Likely finish range"):
    if not forecast.empty:
        finish = forecast[["return_so_far", "p05", "median", "p95", "win_prob"]].copy() * 100
        finish.columns = ["Return now", "Bad case (5%)", "Typical", "Good case (95%)", "Win chance"]
        finish.insert(0, "Student", [student_of[t] for t in finish.index])
        finish.insert(1, "Ticker", finish.index)
        st.dataframe(
            finish,
            hide_index=True,
            use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%+.1f%%") for c in finish.columns[2:6]}
            | {"Win chance": st.column_config.NumberColumn(format="%.1f%%")},
        )

with st.expander("Returns since Oct 20"):
    series = []
    for t in pick_tickers + [BENCHMARK_TICKER]:
        s = snapshot.get(t, {}).get("series")
        if s is not None:
            label = BENCHMARK_NAME if t == BENCHMARK_TICKER else f"{student_of[t]} · {t}"
            series += [{"Date": d, "Pick": label, "Return": v} for d, v in s.items()]
    if series:
        lines = pd.DataFrame(series)
        top_picks = [f"{r.Student} · {r.Ticker}" for r in ranked.head(TOP_N).itertuples()]
        chosen = st.multiselect("Highlight", sorted(lines["Pick"].unique()), default=top_picks, max_selections=TOP_N)
        x = alt.X("Date:T", title=None, axis=alt.Axis(format="%b %d"))
        y = alt.Y("Return:Q", title="Total return", axis=alt.Axis(format="+.0%"))
        chart = alt.layer(
            alt.Chart(lines[~lines["Pick"].isin(chosen + [BENCHMARK_NAME])])
            .mark_line(strokeWidth=1, color="#c3c2b7").encode(x=x, y=y, detail="Pick:N"),
            alt.Chart(lines[lines["Pick"] == BENCHMARK_NAME])
            .mark_line(strokeWidth=2, strokeDash=[6, 4], color=MARKET_GREY).encode(x=x, y=y),
            alt.Chart(lines[lines["Pick"].isin(chosen)])
            .mark_line(strokeWidth=2.5)
            .encode(x=x, y=y, color=alt.Color("Pick:N", title=None, scale=alt.Scale(domain=chosen, range=COLORS[: len(chosen)]),
                                            legend=alt.Legend(orient="top")),
                    tooltip=["Pick:N", alt.Tooltip("Date:T", format="%b %d"), alt.Tooltip("Return:Q", format="+.2%")]),
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)

with st.expander("Quotes & valuation"):
    try:
        quotes = get_quotes(tuple(pick_tickers))
    except Exception:
        quotes = pd.DataFrame()
    if not quotes.empty:
        quotes.insert(0, "Student", quotes["Ticker"].map(student_of))
        quotes["Spread (%)"] = quotes["Spread (%)"] * 100
        quotes["Market cap ($B)"] = quotes["Market cap"] / 1e9
        cols = ["Student", "Ticker", "Price", "Bid", "Ask", "Spread (%)", "P/E (trailing)", "P/E (forward)",
                "Market cap ($B)", "Beta", "52-wk low", "52-wk high", "Sector"]
        money = st.column_config.NumberColumn(format="$%.2f")
        st.dataframe(
            quotes[cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Price": money, "Bid": money, "Ask": money, "52-wk low": money, "52-wk high": money,
                "Spread (%)": st.column_config.NumberColumn(format="%.3f%%"),
                "P/E (trailing)": st.column_config.NumberColumn(format="%.1f"),
                "P/E (forward)": st.column_config.NumberColumn(format="%.1f"),
                "Market cap ($B)": st.column_config.NumberColumn(format="%.1f"),
                "Beta": st.column_config.NumberColumn(format="%.2f"),
            },
        )

with st.expander("Risk (past 12 months)"):
    if not year.empty:
        risk = risk_table(year)
        risk.insert(0, "Student", risk["Ticker"].map(lambda t: student_of.get(t, BENCHMARK_NAME)))
        for c in ["1-year return", "Volatility (yearly)", "Max drawdown", "Best day", "Worst day"]:
            risk[c] = risk[c] * 100
        pct = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(
            risk.sort_values("Volatility (yearly)", ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={
                "1-year return": pct, "Max drawdown": pct, "Best day": pct, "Worst day": pct,
                "Volatility (yearly)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

st.caption("Win chance: 10,000 Monte Carlo simulations replaying real trading days from the past year. Not investment advice.")
