"""MFC Stock Forecast: one-page dashboard.

Run with:
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.config import BENCHMARK_NAME, BENCHMARK_TICKER, COMPETITION_END, DATA_DIR, PRICE_CACHE_SECONDS
from src.data_store import load_picks
from src.earnings import fetch_earnings_events
from src.forecast import DEFAULT_SEED, DEFAULT_SIMS, MODEL_VERSION, compare_models, current_returns, days_left_for, run_forecast
from src.forecast_log import load_log
from src.market_data import fetch_quotes, fetch_year_of_closes
from src.models import MODELS
from src.prices import now_et, snapshot_for_tickers
from src.risk import risk_table

# Colorblind-checked series colors, always used in this order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SHAPES = ["circle", "square", "triangle-up", "diamond", "cross", "triangle-down"]
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
def get_model_comparison(current: pd.Series, year: pd.DataFrame, days_left: int, events: dict | None) -> pd.DataFrame:
    return compare_models(current, year, days_left, n_sims=DEFAULT_SIMS, seed=DEFAULT_SEED, events=events)


@st.cache_data(ttl=6 * 3600)  # earnings dates rarely change
def get_earnings(tickers: tuple[str, ...], today, end) -> dict | None:
    try:
        return fetch_earnings_events(list(tickers), today, end)
    except Exception:
        return None


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

# ---------- Leaderboard data + headline numbers ----------
rows = []
for t in pick_tickers:
    data = snapshot.get(t, {})
    total = data.get("total_return")

    def pct(value):
        return value * 100 if value is not None else None

    rows.append(
        {
            "Student": student_of[t],
            "Company": company_of[t],
            "Ticker": t,
            "Start (Oct 20 open)": data.get("start_price"),
            "Current": data.get("current_price"),
            "Price return": pct(data.get("price_return")),
            "Dividends": pct(data.get("dividend_return")),
            "Return": pct(total),
            "vs S&P 500": (total - spy) * 100 if total is not None and spy is not None else None,
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

# ---------- Tabs ----------
(leaderboard_tab, returns_tab, quotes_tab, perf_tab, mc_tab, models_tab, backtest_tab) = st.tabs(
    ["Leaderboard", "Returns over time", "Quotes & valuation", "1-year performance", "Monte Carlo", "Model comparison",
     "Backtest"]
)

with leaderboard_tab:
    money = st.column_config.NumberColumn(format="$%.2f")
    pct_col = st.column_config.NumberColumn(format="%+.2f%%")
    st.dataframe(
        board,
        hide_index=True,
        use_container_width=True,
        height=35 * (len(board) + 1) + 3,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Start (Oct 20 open)": money,
            "Current": money,
            "Price return": pct_col,
            "Dividends": pct_col,
            "Return": pct_col,
            "vs S&P 500": pct_col,
        },
    )

log = load_log()
official_log = log[log["model"] == MODEL_VERSION] if not log.empty else log

with returns_tab:
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
        x = alt.X("Date:T", title=None, axis=alt.Axis(format="%b %d", tickCount="day"))
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

with quotes_tab:
    try:
        quotes = get_quotes(tuple(pick_tickers))
    except Exception:
        quotes = pd.DataFrame()
    if not quotes.empty:
        quotes.insert(0, "Student", quotes["Ticker"].map(student_of))
        numeric = ["Price", "Bid", "Ask", "Spread (%)", "P/E (trailing)", "P/E (forward)", "Market cap", "Beta",
                   "52-wk low", "52-wk high"]
        quotes[numeric] = quotes[numeric].apply(pd.to_numeric, errors="coerce")  # blanks instead of "None"
        quotes["Spread (%)"] = quotes["Spread (%)"] * 100
        quotes["Market cap ($B)"] = quotes["Market cap"] / 1e9
        cols = ["Student", "Ticker", "Price", "Bid", "Ask", "Spread (%)", "P/E (trailing)", "P/E (forward)",
                "Market cap ($B)", "Beta", "52-wk low", "52-wk high", "Sector"]
        def fmt(spec):
            return lambda v: "—" if pd.isna(v) else spec.format(v)

        shown = quotes[cols].copy()
        for c, spec in {
            "Price": "${:,.2f}", "Bid": "${:,.2f}", "Ask": "${:,.2f}", "52-wk low": "${:,.2f}", "52-wk high": "${:,.2f}",
            "Spread (%)": "{:.3f}%", "P/E (trailing)": "{:.1f}", "P/E (forward)": "{:.1f}",
            "Market cap ($B)": "{:,.1f}", "Beta": "{:.2f}",
        }.items():
            shown[c] = shown[c].map(fmt(spec))
        shown["Sector"] = shown["Sector"].fillna("—")
        st.dataframe(shown, hide_index=True, use_container_width=True, height=35 * (len(shown) + 1) + 3)

with perf_tab:
    if not year.empty:
        available = [t for t in pick_tickers if t in year.columns]
        chosen = st.multiselect(
            "Compare", available, default=[t for t in ranked["Ticker"].head(3)], max_selections=TOP_N,
            format_func=lambda t: f"{student_of[t]} · {t}",
        )
        growth = year / year.apply(lambda col: col.dropna().iloc[0]) * 100
        growth.index.name = "Date"
        long = growth.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Value").dropna()
        x = alt.X("Date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month"))
        y = alt.Y("Value:Q", title="$100 invested a year ago", axis=alt.Axis(format="$,.0f"), scale=alt.Scale(zero=False))
        chart = alt.layer(
            alt.Chart(long[long["Ticker"] == BENCHMARK_TICKER])
            .mark_line(strokeWidth=2, strokeDash=[6, 4], color=MARKET_GREY).encode(x=x, y=y),
            alt.Chart(long[long["Ticker"].isin(chosen)])
            .mark_line(strokeWidth=2)
            .encode(x=x, y=y, color=alt.Color("Ticker:N", title=None, scale=alt.Scale(domain=chosen, range=COLORS[: len(chosen)]),
                                            legend=alt.Legend(orient="top")),
                    tooltip=["Ticker:N", alt.Tooltip("Date:T", format="%b %d, %Y"), alt.Tooltip("Value:Q", format="$,.2f")]),
        ).properties(height=380)
        st.altair_chart(chart, use_container_width=True)
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

with mc_tab:
    if not forecast.empty:
        bars_data = pd.DataFrame(
            {"Pick": [f"{student_of[t]} · {t}" for t in forecast.index], "Win chance": forecast["win_prob"].values}
        )
        bars = alt.Chart(bars_data).mark_bar(color=COLORS[0], cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("Pick:N", sort="-x", title=None),
            x=alt.X("Win chance:Q", title="Win chance", axis=alt.Axis(format=".0%", tickCount=5)),
            tooltip=["Pick:N", alt.Tooltip("Win chance:Q", format=".1%")],
        )
        labels = bars.mark_text(align="left", dx=4, color=MARKET_GREY).encode(text=alt.Text("Win chance:Q", format=".1%"))
        st.altair_chart((bars + labels).properties(height=26 * len(bars_data) + 40), use_container_width=True)
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
    st.subheader("Win chance over time")
    if official_log["forecast_date"].nunique() < 2 if not official_log.empty else True:
        st.caption("Needs at least 2 logged days.")
    else:
        official_log = official_log.assign(Pick=official_log["student"] + " · " + official_log["ticker"])
        latest_day = official_log["forecast_date"].max()
        default = (
            official_log[official_log["forecast_date"] == latest_day]
            .sort_values("win_prob", ascending=False)["Pick"].head(TOP_N).tolist()
        )
        follow = st.multiselect("Picks", sorted(official_log["Pick"].unique()), default=default, max_selections=TOP_N)
        hist = official_log[official_log["Pick"].isin(follow)].copy()
        hist["Day"] = pd.to_datetime(hist["forecast_date"]).dt.strftime("%b %d")
        day_order = [pd.Timestamp(d).strftime("%b %d") for d in sorted(hist["forecast_date"].unique())]
        chart = (
            alt.Chart(hist)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True))
            .encode(
                x=alt.X("Day:O", sort=day_order, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("win_prob:Q", title="Win chance", axis=alt.Axis(format=".0%")),
                color=alt.Color("Pick:N", title=None, scale=alt.Scale(domain=follow, range=COLORS[: len(follow)]),
                                legend=alt.Legend(orient="top")),
                tooltip=["Pick:N", "Day:N",
                         alt.Tooltip("win_prob:Q", title="Win chance", format=".1%")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

with models_tab:
    if year.empty:
        st.write("No price history.")
    else:
        events = get_earnings(tuple(pick_tickers), now.date(), COMPETITION_END.date())
        odds = get_model_comparison(current, year, days_left, events)
        earnings_day = {t: e["next_reaction"] for t, e in (events or {}).get("tickers", {}).items() if e.get("next_reaction")}
        favorites = pd.DataFrame(
            [
                {
                    "Model": MODELS[k][0],
                    "Favorite": f"{student_of[odds[k].idxmax()]} · {odds[k].idxmax()}",
                    "Win chance": odds[k].max() * 100,
                    "Top 3 combined": odds[k].nlargest(3).sum() * 100,
                }
                for k in MODELS
            ]
        )
        st.dataframe(
            favorites,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Win chance": st.column_config.NumberColumn(format="%.1f%%"),
                "Top 3 combined": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )
        names = [MODELS[k][0] for k in MODELS]
        table = odds.copy() * 100
        table.columns = names
        table["Spread"] = table.max(axis=1) - table.min(axis=1)
        table.insert(0, "Student", [student_of[t] for t in table.index])
        table.insert(1, "Ticker", table.index)
        table["Earnings day"] = [str(earnings_day.get(t, "—")) for t in table.index]
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


# ---------- Backtest ----------
with backtest_tab:
    summary_path, calib_path = DATA_DIR / "backtest_summary.csv", DATA_DIR / "backtest_calibration.csv"
    results_path = DATA_DIR / "backtest_results.csv"
    if not summary_path.exists():
        st.caption("No backtest yet. Run `python backtest.py` in the project folder.")
    else:
        summary = pd.read_csv(summary_path)
        results_bt = pd.read_csv(results_path)
        names = {k: MODELS[k][0] for k in MODELS}
        summary["Model"] = summary["model"].map(names)
        k1, k2, k3 = st.columns(3)
        k1.metric("Past contests", f"{results_bt['window_start'].nunique()}")
        k2.metric("From", str(results_bt["window_start"].min()))
        k3.metric("To", str(results_bt["window_end"].max()))

        table = summary.rename(columns={
            "days_left": "Days left", "windows": "Contests", "avg_winner_prob": "Chance given to winner",
            "blind_guess_prob": "Blind guess", "favorite_won": "Favorite won", "leader_won": "Leader won",
            "brier_skill": "Skill vs blind guess", "log_loss": "Log loss",
        })[["Days left", "Model", "Contests", "Chance given to winner", "Blind guess", "Favorite won", "Leader won",
            "Skill vs blind guess", "Log loss"]]
        for c in ["Chance given to winner", "Blind guess", "Favorite won", "Skill vs blind guess"]:
            table[c] = table[c] * 100
        # No leader exists at the start of a contest, so show a dash there instead of "None".
        table["Leader won"] = table["Leader won"].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}")
        pct1 = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            height=35 * (len(table) + 1) + 3,
            column_config={
                "Chance given to winner": pct1, "Blind guess": pct1, "Skill vs blind guess": pct1,
                "Favorite won": st.column_config.NumberColumn(format="%.0f%%"),
                "Log loss": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.subheader("Calibration")
        calib = pd.read_csv(calib_path)
        calib["Model"] = calib["model"].map(names)
        model_names = [names[k] for k in MODELS if names[k] in set(calib["Model"])]
        chosen_model = st.selectbox("Model", model_names, index=0)
        points = calib[calib["Model"] == chosen_model].copy()
        top = float(max(points["mean_pred"].max(), points["observed"].max(), 0.05)) * 1.1
        axis = alt.Axis(format=".0%", tickCount=6)
        diagonal = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(
            color="#c3c2b7", strokeDash=[6, 4]
        ).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, top])), y=alt.Y("y:Q", scale=alt.Scale(domain=[0, top])))
        dots = alt.Chart(points).mark_circle(color=COLORS[0], opacity=0.9, stroke="white", strokeWidth=1).encode(
            x=alt.X("mean_pred:Q", title="Predicted chance of winning", axis=axis, scale=alt.Scale(domain=[0, top])),
            y=alt.Y("observed:Q", title="How often those picks actually won", axis=axis, scale=alt.Scale(domain=[0, top])),
            size=alt.Size("count:Q", title="Forecasts", scale=alt.Scale(range=[60, 600]), legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("bin:N", title="Range"), alt.Tooltip("mean_pred:Q", title="Predicted", format=".1%"),
                     alt.Tooltip("observed:Q", title="Actually won", format=".1%"), alt.Tooltip("count:Q", title="Forecasts")],
        )
        st.altair_chart((diagonal + dots).properties(height=420), use_container_width=True)
        shown = points[["bin", "mean_pred", "observed", "count"]].rename(
            columns={"bin": "Range", "mean_pred": "Predicted", "observed": "Actually won", "count": "Forecasts"}
        )
        shown["Predicted"] *= 100
        shown["Actually won"] *= 100
        st.dataframe(shown, hide_index=True, use_container_width=True,
                     column_config={"Predicted": pct1, "Actually won": pct1})
