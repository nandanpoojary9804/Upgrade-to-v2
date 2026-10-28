"""
NSE terminal: Streamlit front end
=================================
Three tabs: Score breakdown, Screener, Alerts.

This app imports the analysis code in nse_analysis_api.py directly, so on
Streamlit Community Cloud you deploy ONE app and do not need a separate
FastAPI server. Keep nse_analysis_api.py in the same folder.

Run locally:   streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st

import nse_analysis_api as api

st.set_page_config(page_title="NSE terminal", page_icon="📈", layout="wide")

DEFAULT_WATCHLIST = "RELIANCE, TCS, INFY, HDFCBANK, SBIN"


# ── helpers ──────────────────────────────────────────────────────────────

def safe(fn, *args, **kwargs):
    """Run fn and return (result, error_message). Never raises into the UI."""
    try:
        return fn(*args, **kwargs), None
    except api.HTTPException as e:
        return None, str(e.detail)
    except Exception as e:  # network hiccups, Yahoo rate limits, etc.
        return None, f"Something went wrong: {e}"


def fmt(v, d=2, suffix=""):
    return "–" if v is None else f"{v:,.{d}f}{suffix}"


@st.cache_data(ttl=600, show_spinner=False)
def load_analysis(ticker: str):
    return api.sanitize(api.build_analysis(ticker))


@st.cache_data(ttl=600, show_spinner=False)
def load_price_frame(ticker: str):
    close = api.get_history(ticker)["Close"]
    df = pd.DataFrame({"Price": close, "50-day average": close.rolling(50).mean(),
                       "200-day average": close.rolling(200).mean()})
    return df.tail(252)


@st.cache_data(ttl=600, show_spinner=False)
def run_screen(max_rsi, min_score, above, min_roe, max_de, max_pe):
    return api.screen(
        universe="nifty50", symbols=None,
        min_rsi=None, max_rsi=max_rsi, min_score=min_score, min_adx=None,
        min_pct_52w_high=None, min_volume_ratio=None,
        above_ma200={"Above": True, "Below": False}.get(above), ma_trend=None,
        min_roe=min_roe, max_de=max_de, max_pe=max_pe,
        min_earnings_growth=None, min_piotroski=None, limit=30)


@st.cache_data(ttl=300, show_spinner=False)
def run_alerts(symbols: str, lookback: int):
    return api.alerts(symbols, lookback)


def breakdown_table(score):
    rows = []
    for b in score["breakdown"]:
        has = b["points"] is not None
        rows.append({
            "Factor": b["factor"].replace("_", " ").capitalize(),
            "Strength": (b["points"] / b["max"] * 100) if has else None,
            "Points": f'{b["points"]}/{b["max"]}' if has else "no data",
            "Why": b.get("note") or ("" if has else "No data available"),
        })
    return pd.DataFrame(rows)


def render_breakdown(title, score):
    st.subheader(f'{title}: {score["rating"]} ({score["score"]})')
    if score.get("data_coverage_pct") is not None:
        st.caption(f'Data coverage {score["data_coverage_pct"]}%')
    st.dataframe(
        breakdown_table(score), hide_index=True,
        column_config={"Strength": st.column_config.ProgressColumn(
            "Strength", min_value=0, max_value=100, format="%d%%")})


def render_analysis(d):
    t = d["technical"]
    f = d.get("fundamentals") or {}
    r = d.get("risk") or {}
    lv = d.get("levels") or {}
    rs = (d.get("relative_strength") or {}).get("periods", {}).get("1y")

    st.header(d["company_name"])
    st.caption(d["symbol"] + (f' in {d["sector"]}' if d.get("sector") else ""))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f'₹{fmt(t["current_price"])}', f'{fmt(t["day_change_pct"])}%' if t.get("day_change_pct") is not None else None)
    c2.metric("Overall score", fmt(d["overall"]["score"], 0), d["overall"]["rating"], delta_color="off")
    c3.metric("RSI (14)", fmt(t["rsi_14"], 1))
    c4.metric("50/200-day trend", t["ma_trend"] or "–")
    st.caption("Overall basis: " + d["overall"]["basis"])

    if d.get("corporate_action_note"):
        st.warning(d["corporate_action_note"])

    price_df, err = safe(load_price_frame, d["symbol"])
    if price_df is not None:
        st.line_chart(price_df)

    left, right = st.columns(2)
    with left:
        render_breakdown("Technical score", d["technical_score"])
    with right:
        if d.get("fundamental_score"):
            render_breakdown("Fundamental score", d["fundamental_score"])

    a, b, c = st.columns(3)
    with a:
        st.subheader("Price levels")
        st.write(f'Support (20 days): ₹{fmt(lv.get("support_20d"))}')
        st.write(f'Resistance (20 days): ₹{fmt(lv.get("resistance_20d"))}')
        st.write(f'Pivot: ₹{fmt((lv.get("pivots") or {}).get("pivot"))}')
        st.write(f'ADX (14): {fmt(t["adx_14"], 1)}')
        st.write(f'Near 52-week high: {fmt(t["pct_of_52w_high"], 1, "%")}')
        st.write(f'Volume vs 20-day average: {fmt(t["volume_ratio_20d"], 2, "x")}')
        st.write("Candle pattern: " + (", ".join(d.get("candlestick_patterns") or []) or "none"))
    with b:
        st.subheader("Fundamentals")
        st.write(f'P/E: {fmt(f.get("pe_ratio"), 1)}   P/B: {fmt(f.get("pb_ratio"), 1)}')
        st.write(f'ROE (trailing 12 months): {fmt(f.get("roe_ttm_pct"), 1, "%")}')
        st.write(f'Debt/equity: {fmt(f.get("debt_to_equity"))}')
        st.write(f'Earnings growth YoY: {fmt(f.get("yoy_earnings_growth_pct"), 1, "%")}')
        st.write(f'Net margin: {fmt(f.get("net_margin_pct"), 1, "%")}')
        st.write(f'Dividend yield: {fmt(f.get("dividend_yield_pct"), 2, "%")}')
        pf = f.get("piotroski_f_score")
        st.write("Piotroski F-score: " + ("–" if pf is None else f"{pf} of 9"))
    with c:
        st.subheader("Risk")
        st.write(f'Beta vs Nifty: {fmt(r.get("beta"))}')
        st.write(f'Volatility (annual): {fmt(r.get("annualized_volatility_pct"), 1, "%")}')
        st.write(f'Sharpe ratio: {fmt(r.get("sharpe_ratio"))}')
        st.write(f'Max drawdown: {fmt(r.get("max_drawdown_pct"), 1, "%")}')
        st.write(f'1-day VaR (95%): {fmt(r.get("var_95_daily_pct"), 2, "%")}')
        st.write("1-year return vs Nifty: " + (f'{rs["excess_return_pct"]:+.1f}%' if rs else "–"))

    trend = f.get("quarterly_trend") or []
    if trend:
        with st.expander("Quarterly trend (last quarters, ₹ crore)"):
            st.dataframe(pd.DataFrame(trend).rename(columns={
                "quarter": "Quarter", "revenue_cr": "Revenue", "net_income_cr": "Net income",
                "net_margin_pct": "Net margin %"}), hide_index=True)


def analyze_and_render(symbol: str):
    ticker = api.to_ticker(symbol)
    with st.spinner(f"Analyzing {ticker}…"):
        data, err = safe(load_analysis, ticker)
    if err:
        st.error(err)
    else:
        render_analysis(data)


# ── layout ───────────────────────────────────────────────────────────────

st.title("NSE terminal")

with st.sidebar:
    st.markdown("**About**")
    st.caption("Scores and signals are rule-based indicators for research and education, "
               "not investment advice. Prices come from Yahoo Finance and can be delayed.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        api._cache.clear()
        st.rerun()

tab_score, tab_screen, tab_alerts = st.tabs(["Score breakdown", "Screener", "Alerts"])

# ── tab 1: score breakdown ──
with tab_score:
    with st.form("score_form"):
        sym = st.text_input("NSE symbol", placeholder="e.g. RELIANCE")
        go = st.form_submit_button("Analyze")
    if go and sym.strip():
        st.session_state["score_symbol"] = sym.strip()
    if st.session_state.get("score_symbol"):
        analyze_and_render(st.session_state["score_symbol"])
    else:
        st.info("Enter an NSE symbol to see exactly how its score is built.")

# ── tab 2: screener ──
with tab_screen:
    st.caption("Runs across the built-in Nifty-50-style stock list. Fundamental filters "
               "download company statements, so the first run can take a minute.")
    with st.form("screen_form"):
        c1, c2, c3 = st.columns(3)
        max_rsi = c1.number_input("Max RSI", value=None, min_value=0.0, max_value=100.0, placeholder="e.g. 35")
        min_score = c2.number_input("Min technical score", value=None, min_value=0.0, max_value=100.0, placeholder="e.g. 60")
        above = c3.selectbox("Price vs 200-day average", ["Any", "Above", "Below"])
        c4, c5, c6 = st.columns(3)
        min_roe = c4.number_input("Min ROE %", value=None, placeholder="e.g. 15")
        max_de = c5.number_input("Max debt/equity", value=None, min_value=0.0, placeholder="e.g. 1")
        max_pe = c6.number_input("Max P/E", value=None, min_value=0.0, placeholder="e.g. 30")
        run = st.form_submit_button("Run screen")
    if run:
        with st.spinner("Screening stocks…"):
            res, err = safe(run_screen, max_rsi, min_score, above, min_roe, max_de, max_pe)
        st.session_state["screen_result"] = (res, err)

    if "screen_result" in st.session_state:
        res, err = st.session_state["screen_result"]
        if err:
            st.error(err)
        elif not res["matches"]:
            st.warning(f'No stocks matched. Loosen a filter and try again. '
                       f'({res["data_available"]} of {res["universe_size"]} stocks had data.)')
        else:
            st.write(f'**{res["match_count"]}** of {res["universe_size"]} stocks matched.')
            df = pd.DataFrame(res["matches"])
            df["symbol"] = df["symbol"].str.replace(".NS", "", regex=False)
            keep = [c for c in ["symbol", "company_name", "sector", "price", "rsi_14", "ma_trend",
                                "technical_score", "rating", "roe_pct", "pe_ratio", "debt_to_equity"] if c in df.columns]
            st.dataframe(df[keep].rename(columns={
                "symbol": "Symbol", "company_name": "Company", "sector": "Sector", "price": "Price",
                "rsi_14": "RSI", "ma_trend": "Trend", "technical_score": "Score", "rating": "Rating",
                "roe_pct": "ROE %", "pe_ratio": "P/E", "debt_to_equity": "D/E"}), hide_index=True)
            pick = st.selectbox("See the score breakdown for", ["Choose a stock"] + list(df["symbol"]))
            if pick != "Choose a stock":
                analyze_and_render(pick)

# ── tab 3: alerts ──
with tab_alerts:
    with st.form("alerts_form"):
        watch = st.text_area("Watchlist (comma-separated)", value=st.session_state.get("watchlist", DEFAULT_WATCHLIST), height=70)
        look = st.selectbox("Look back", ["Today", "2 days", "5 days"])
        check = st.form_submit_button("Check alerts")
    if check:
        st.session_state["watchlist"] = watch
        days = {"Today": 1, "2 days": 2, "5 days": 5}[look]
        with st.spinner("Checking alerts…"):
            res, err = safe(run_alerts, watch.replace(" ", ""), days)
        st.session_state["alerts_result"] = (res, err)

    if "alerts_result" in st.session_state:
        res, err = st.session_state["alerts_result"]
        if err:
            st.error(err)
        else:
            n_with = sum(1 for s in res["by_symbol"] if s.get("alert_count"))
            st.write(f'**{res["total_alerts"]}** {"alert" if res["total_alerts"] == 1 else "alerts"} '
                     f'across {n_with} of {len(res["by_symbol"])} stocks.')
            for s in res["by_symbol"]:
                if s.get("alert_count"):
                    with st.container(border=True):
                        st.markdown(f'**{s["symbol"].replace(".NS", "")}**  ₹{fmt(s["price"])}')
                        for a in s["alerts"]:
                            (st.warning if a["severity"] == "warning" else st.info)(a["message"])
            quiet = [s["symbol"].replace(".NS", "") for s in res["by_symbol"] if s.get("alert_count") == 0]
            if quiet:
                st.caption("No alerts for " + ", ".join(quiet) + ".")
            failed = [s["symbol"].replace(".NS", "") for s in res["by_symbol"] if s.get("error")]
            if failed:
                st.error("No data for " + ", ".join(failed) + ". Check the symbols.")
