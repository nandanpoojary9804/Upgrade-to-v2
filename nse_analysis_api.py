"""
NSE Stock Analysis API  (v2)  — for n8n / terminal-app integration
================================================================================
Same service as before, extended with scoring, valuation, risk, screening,
backtesting, alerts, portfolio tracking, quant tools and NSE-specific data.

ALL ORIGINAL ENDPOINTS AND FIELDS STILL WORK. New data is added alongside them.

REQUIREMENTS
------------
    pip install -r requirements.txt
    (fastapi uvicorn yfinance pandas numpy requests statsmodels)

    statsmodels is optional: only /quant and /pairs need it for p-values.

RUN LOCALLY
-----------
    uvicorn nse_analysis_api:app --reload --port 8000
    Interactive docs:  http://127.0.0.1:8000/docs

ENDPOINTS
---------
  Core (original, now enriched)
    GET  /analyze/{symbol}        technical + fundamentals + scores + risk + levels
    GET  /chart/{symbol}          OHLCV + indicator series (now with Bollinger/VWAP)
    GET  /watchlist?symbols=      multi-stock snapshot (parallel, up to 25 symbols)

  New: analysis
    GET  /risk/{symbol}           beta, alpha, Sharpe, Sortino, drawdown, VaR
    GET  /compare?symbols=        side-by-side peer comparison with ranking
    GET  /correlation?symbols=    correlation matrix of daily returns
    GET  /sectors                 sector heatmap for the Nifty-50-style universe
    GET  /screen                  filter the universe on technical + fundamental rules
    GET  /alerts?symbols=         triggered signals (built for n8n schedules)
    GET  /backtest/{symbol}       golden_cross | rsi | macd | bollinger strategies
    POST /portfolio               P&L, weights, portfolio beta/volatility/VaR

  New: quant
    GET  /quant/{symbol}          ADF test, Hurst exponent, half-life, ACF, vol forecast
    GET  /pairs?a=&b=             cointegration + spread z-score for a pair

  New: NSE-specific data (best effort, see NOTE in that section)
    GET  /nse/option-chain/{symbol}   PCR, max pain, OI support/resistance
    GET  /nse/fii-dii                 FII / DII cash-market activity
    GET  /nse/announcements/{symbol}  corporate announcements
    GET  /nse/bulk-deals              bulk / block deals
    GET  /nse/delivery/{symbol}       delivery percentage

  Utility
    POST /cache/clear

DISCLAIMER: scores and signals are rule-based indicators for research and
education. They are not investment advice.
"""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # optional: only used for ADF / cointegration p-values
    from statsmodels.tsa.stattools import adfuller, coint
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

app = FastAPI(
    title="NSE Stock Analysis API",
    version="2.0.0",
    description="Technical, fundamental, risk, screening, backtesting and NSE data for NSE-listed stocks.",
)

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ────────────────────────────────────────────────────────────────

BENCHMARK = "^NSEI"          # Nifty 50 index on Yahoo Finance
RISK_FREE_RATE = 0.07        # approx. Indian 10-year G-sec yield; change as needed
TRADING_DAYS = 252
HISTORY_TTL = 600            # seconds: price history cache (10 min)
FUNDAMENTALS_TTL = 6 * 3600  # seconds: statements change rarely (6 h)
NSE_TTL = 120                # seconds: NSE responses cache
MAX_WORKERS = 8              # parallel Yahoo requests
MAX_WATCHLIST = 25
MAX_UNIVERSE = 60

# Nifty-50-style universe with names and sectors. Index membership changes over
# time, so edit this dict whenever you want to add or remove stocks.
UNIVERSE: Dict[str, tuple] = {
    "RELIANCE": ("Reliance Industries", "Energy"),
    "TCS": ("Tata Consultancy Services", "IT"),
    "HDFCBANK": ("HDFC Bank", "Financial Services"),
    "ICICIBANK": ("ICICI Bank", "Financial Services"),
    "INFY": ("Infosys", "IT"),
    "HINDUNILVR": ("Hindustan Unilever", "FMCG"),
    "ITC": ("ITC", "FMCG"),
    "SBIN": ("State Bank of India", "Financial Services"),
    "BHARTIARTL": ("Bharti Airtel", "Telecom"),
    "KOTAKBANK": ("Kotak Mahindra Bank", "Financial Services"),
    "LT": ("Larsen & Toubro", "Infrastructure"),
    "AXISBANK": ("Axis Bank", "Financial Services"),
    "ASIANPAINT": ("Asian Paints", "Consumer Durables"),
    "MARUTI": ("Maruti Suzuki", "Automobile"),
    "SUNPHARMA": ("Sun Pharmaceutical", "Healthcare"),
    "TITAN": ("Titan Company", "Consumer Durables"),
    "ULTRACEMCO": ("UltraTech Cement", "Materials"),
    "BAJFINANCE": ("Bajaj Finance", "Financial Services"),
    "NESTLEIND": ("Nestle India", "FMCG"),
    "WIPRO": ("Wipro", "IT"),
    "HCLTECH": ("HCL Technologies", "IT"),
    "M&M": ("Mahindra & Mahindra", "Automobile"),
    "NTPC": ("NTPC", "Power"),
    "POWERGRID": ("Power Grid Corporation", "Power"),
    "ONGC": ("Oil & Natural Gas Corporation", "Energy"),
    "TATASTEEL": ("Tata Steel", "Metals"),
    "JSWSTEEL": ("JSW Steel", "Metals"),
    "TECHM": ("Tech Mahindra", "IT"),
    "ADANIENT": ("Adani Enterprises", "Diversified"),
    "ADANIPORTS": ("Adani Ports & SEZ", "Infrastructure"),
    "COALINDIA": ("Coal India", "Energy"),
    "BAJAJFINSV": ("Bajaj Finserv", "Financial Services"),
    "INDUSINDBK": ("IndusInd Bank", "Financial Services"),
    "HINDALCO": ("Hindalco Industries", "Metals"),
    "GRASIM": ("Grasim Industries", "Materials"),
    "CIPLA": ("Cipla", "Healthcare"),
    "DRREDDY": ("Dr. Reddy's Laboratories", "Healthcare"),
    "EICHERMOT": ("Eicher Motors", "Automobile"),
    "APOLLOHOSP": ("Apollo Hospitals", "Healthcare"),
    "BRITANNIA": ("Britannia Industries", "FMCG"),
    "HEROMOTOCO": ("Hero MotoCorp", "Automobile"),
    "BPCL": ("Bharat Petroleum", "Energy"),
    "TATACONSUMER": ("Tata Consumer Products", "FMCG"),
    "SBILIFE": ("SBI Life Insurance", "Financial Services"),
    "HDFCLIFE": ("HDFC Life Insurance", "Financial Services"),
    "DIVISLAB": ("Divi's Laboratories", "Healthcare"),
    "TRENT": ("Trent", "Consumer Services"),
    "BAJAJ-AUTO": ("Bajaj Auto", "Automobile"),
    "SHRIRAMFIN": ("Shriram Finance", "Financial Services"),
}

# ── CACHE + DATA ACCESS ───────────────────────────────────────────────────


class TTLCache:
    """Tiny thread-safe in-memory cache with per-entry expiry."""

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.time():
                del self._data[key]
                return None
            return value

    def set(self, key, value, ttl):
        with self._lock:
            self._data[key] = (time.time() + ttl, value)

    def clear(self):
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n


_cache = TTLCache()


def to_ticker(symbol: str) -> str:
    """RELIANCE -> RELIANCE.NS ; leaves ^NSEI, *.NS and *.BO untouched."""
    s = symbol.upper().strip()
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"


def base_symbol(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith((".NS", ".BO")) else ticker


def get_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Cached price history. Returns an empty DataFrame if nothing is found."""
    key = ("hist", ticker, period)
    cached = _cache.get(key)
    if cached is not None:
        return cached.copy()

    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is None or hist.empty:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    hist.index = idx.normalize()  # date-only index so stock and benchmark align
    _cache.set(key, hist, HISTORY_TTL)
    return hist.copy()


def get_histories(tickers: List[str], period: str = "2y") -> Dict[str, Optional[pd.DataFrame]]:
    """Download several histories in parallel. Failed/empty tickers map to None."""
    out: Dict[str, Optional[pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {t: pool.submit(get_history, t, period) for t in tickers}
        for t, fut in futures.items():
            try:
                h = fut.result()
                out[t] = h if h is not None and not h.empty else None
            except Exception:
                out[t] = None
    return out


def get_benchmark_close(period: str = "2y", benchmark: str = BENCHMARK) -> Optional[pd.Series]:
    try:
        h = get_history(benchmark, period)
        return h["Close"] if not h.empty else None
    except Exception:
        return None


def require_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    hist = get_history(ticker, period)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")
    return hist


def sanitize(obj):
    """Recursively convert numpy types, NaN and inf into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return obj


def parse_symbols(symbols: str, cap: int, minimum: int = 1) -> List[str]:
    out, seen = [], set()
    for s in symbols.split(","):
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    out = out[:cap]
    if len(out) < minimum:
        raise HTTPException(status_code=400, detail=f"Provide at least {minimum} symbol(s), comma-separated.")
    return out


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


# ── INDICATOR FUNCTIONS (originals, unchanged) ────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug
    we caught earlier today where fixed-position matching broke on
    stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


# ── NEW INDICATOR FUNCTIONS ───────────────────────────────────────────────

def calculate_atr(high, low, close, period=14):
    """Average True Range (Wilder smoothing): typical daily price range in rupees."""
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_obv(close, volume):
    """On-Balance Volume: cumulative volume signed by the day's price direction."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calculate_vwap(high, low, close, volume, window=20):
    """Rolling VWAP over `window` days (daily data has no true intraday VWAP)."""
    typical = (high + low + close) / 3
    vol_sum = volume.rolling(window).sum()
    return (typical * volume).rolling(window).sum() / vol_sum.replace(0, np.nan)


def calculate_pivot_points(high, low, close):
    """Classic floor-trader pivots from the latest completed bar."""
    h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    p = (h + l + c) / 3
    return {
        "pivot": safe_round(p),
        "r1": safe_round(2 * p - l), "r2": safe_round(p + (h - l)), "r3": safe_round(h + 2 * (p - l)),
        "s1": safe_round(2 * p - h), "s2": safe_round(p - (h - l)), "s3": safe_round(l - 2 * (h - p)),
    }


def calculate_support_resistance(high, low):
    return {
        "support_20d": safe_round(low.tail(20).min()),
        "resistance_20d": safe_round(high.tail(20).max()),
        "support_60d": safe_round(low.tail(60).min()),
        "resistance_60d": safe_round(high.tail(60).max()),
    }


def detect_candlestick_patterns(open_, high, low, close):
    """Flags simple single/two-candle patterns on the latest bar."""
    patterns = []
    if len(close) < 2:
        return patterns
    o, h, l, c = (float(x.iloc[-1]) for x in (open_, high, low, close))
    po, pc = float(open_.iloc[-2]), float(close.iloc[-2])
    rng = h - l
    if rng <= 0:
        return patterns
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    if body / rng < 0.1:
        patterns.append("doji")
    elif lower >= 2 * body and upper <= 0.5 * body:
        patterns.append("hammer")
    elif upper >= 2 * body and lower <= 0.5 * body:
        patterns.append("shooting_star")

    if pc < po and c > o and c >= po and o <= pc:
        patterns.append("bullish_engulfing")
    if pc > po and c < o and c <= po and o >= pc:
        patterns.append("bearish_engulfing")
    return patterns


# ── TECHNICAL SNAPSHOT (original fields + new ones) ───────────────────────

def get_technical_snapshot(hist):
    """The core technical calculation, shared by /analyze, /watchlist, /compare,
    /screen and /alerts. Original keys are unchanged; new keys are additive."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    atr = calculate_atr(high, low, close)
    obv = calculate_obv(close, volume)
    obv_sma = obv.rolling(20).mean()
    vol_avg20 = volume.rolling(20).mean()
    vwap = calculate_vwap(high, low, close, volume)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    last_close = close.iloc[-1]
    obv_trend = None
    if not pd.isna(obv_sma.iloc[-1]):
        obv_trend = "rising" if obv.iloc[-1] > obv_sma.iloc[-1] else "falling"

    volume_ratio = None
    if not pd.isna(vol_avg20.iloc[-1]) and vol_avg20.iloc[-1] > 0:
        volume_ratio = volume.iloc[-1] / vol_avg20.iloc[-1]

    ann_vol = daily_pct_change.tail(TRADING_DAYS).std() * math.sqrt(TRADING_DAYS) * 100

    technical = {
        # ── original fields ──
        "current_price": safe_round(last_close),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
        # ── new fields ──
        "day_change_pct": safe_round(daily_pct_change.iloc[-1] * 100),
        "atr_14": safe_round(atr.iloc[-1]),
        "atr_pct_of_price": safe_round(atr.iloc[-1] / last_close * 100) if last_close else None,
        "annualized_volatility_pct": safe_round(ann_vol),
        "volume_ratio_20d": safe_round(volume_ratio),
        "obv_trend": obv_trend,
        "vwap_20d": safe_round(vwap.iloc[-1]),
        "price_vs_vwap_pct": safe_round((last_close / vwap.iloc[-1] - 1) * 100) if not pd.isna(vwap.iloc[-1]) else None,
    }
    return technical, corporate_action_flag


def get_levels_and_patterns(hist):
    levels = calculate_support_resistance(hist["High"], hist["Low"])
    levels["pivots"] = calculate_pivot_points(hist["High"], hist["Low"], hist["Close"])
    patterns = detect_candlestick_patterns(hist["Open"], hist["High"], hist["Low"], hist["Close"])
    return levels, patterns


# ── SIGNAL SCORING (technical + fundamental) ──────────────────────────────

def rating_label(score):
    if score is None:
        return None
    if score >= 75:
        return "Strong Buy"
    if score >= 60:
        return "Buy"
    if score >= 45:
        return "Hold"
    if score >= 30:
        return "Weak"
    return "Avoid"


def compute_signal_score(tech):
    """Transparent 0-100 technical score. Every point is explained in `breakdown`.

    Trend 30 · Momentum 30 · Trend strength 10 · Price position 15 · Volume 15
    """
    parts = []

    def add(name, max_pts, pts, note):
        parts.append({"factor": name, "points": pts, "max": max_pts, "note": note})

    price, ma50, ma200 = tech.get("current_price"), tech.get("ma_50"), tech.get("ma_200")
    trend = tech.get("ma_trend")

    # TREND (30)
    add("ma_cross", 15, 15 if trend == "bullish" else 0 if trend == "bearish" else 7,
        f"50DMA vs 200DMA: {trend or 'insufficient data'}")
    add("price_vs_200dma", 8, 8 if (price and ma200 and price > ma200) else 0,
        "price above 200DMA" if (price and ma200 and price > ma200) else "price below 200DMA or no data")
    add("price_vs_50dma", 7, 7 if (price and ma50 and price > ma50) else 0,
        "price above 50DMA" if (price and ma50 and price > ma50) else "price below 50DMA or no data")

    # MOMENTUM (30)
    rsi = tech.get("rsi_14")
    if rsi is None:
        rsi_pts, rsi_note = 6, "no RSI"
    elif rsi < 30:
        rsi_pts, rsi_note = 8, "oversold (<30): bounce potential"
    elif rsi < 45:
        rsi_pts, rsi_note = 6, "weak but not oversold"
    elif rsi <= 65:
        rsi_pts, rsi_note = 12, "healthy momentum zone"
    elif rsi <= 70:
        rsi_pts, rsi_note = 9, "strong, nearing overbought"
    else:
        rsi_pts, rsi_note = 4, "overbought (>70): pullback risk"
    add("rsi", 12, rsi_pts, rsi_note)

    macd_h = tech.get("macd_histogram")
    add("macd", 10, 10 if (macd_h is not None and macd_h > 0) else 0,
        "MACD above signal" if (macd_h is not None and macd_h > 0) else "MACD below signal or no data")
    roc = tech.get("roc_12")
    add("roc", 8, 8 if (roc is not None and roc > 0) else 0,
        "positive 12-day rate of change" if (roc is not None and roc > 0) else "non-positive rate of change or no data")

    # TREND STRENGTH (10)
    adx = tech.get("adx_14")
    if adx is None:
        adx_pts, adx_note = 5, "no ADX"
    elif adx >= 25:
        adx_pts, adx_note = (10, "strong uptrend") if trend == "bullish" else (0, "strong downtrend")
    else:
        adx_pts, adx_note = 5, "weak/no trend (ADX<25)"
    add("adx", 10, adx_pts, adx_note)

    # PRICE POSITION (15)
    pb = tech.get("bollinger_percent_b")
    if pb is None:
        pb_pts, pb_note = 5, "no Bollinger data"
    elif pb > 1:
        pb_pts, pb_note = 3, "above upper band: stretched"
    elif pb > 0.8:
        pb_pts, pb_note = 4, "near upper band"
    elif pb >= 0.2:
        pb_pts, pb_note = 8, "inside bands"
    else:
        pb_pts, pb_note = 6, "near/below lower band: oversold zone"
    add("bollinger", 8, pb_pts, pb_note)

    h52 = tech.get("pct_of_52w_high")
    if h52 is None:
        h_pts, h_note = 3, "no 52w data"
    elif h52 >= 90:
        h_pts, h_note = 7, "within 10% of 52w high"
    elif h52 >= 75:
        h_pts, h_note = 5, "within 25% of 52w high"
    elif h52 >= 60:
        h_pts, h_note = 3, "25-40% below 52w high"
    else:
        h_pts, h_note = 1, ">40% below 52w high"
    add("52w_high_proximity", 7, h_pts, h_note)

    # VOLUME (15)
    vr, chg = tech.get("volume_ratio_20d"), tech.get("day_change_pct")
    if vr is None:
        v_pts, v_note = 4, "no volume data"
    elif vr >= 1.5 and chg is not None and chg > 0:
        v_pts, v_note = 8, "high-volume up day: accumulation"
    elif vr >= 1:
        v_pts, v_note = 5, "volume at/above average"
    else:
        v_pts, v_note = 3, "volume below average"
    add("volume", 8, v_pts, v_note)

    obv = tech.get("obv_trend")
    add("obv", 7, 7 if obv == "rising" else 2 if obv == "falling" else 3,
        f"OBV {obv}" if obv else "no OBV data")

    total = sum(p["points"] for p in parts)
    return {"score": total, "rating": rating_label(total), "breakdown": parts}


def compute_fundamental_score(f, is_financial=False):
    """0-100 fundamental score. Missing data earns half marks and lowers `data_coverage_pct`.
    For banks/NBFCs, leverage and interest-coverage items are skipped (not meaningful)."""
    items = []

    def add(name, max_pts, pts):
        items.append((name, max_pts, pts))

    def band(v, tiers, floor):
        """tiers = [(threshold, pts), ...] checked in order with v > threshold."""
        if v is None:
            return None
        for threshold, pts in tiers:
            if v > threshold:
                return pts
        return floor

    add("earnings_growth", 20, band(f.get("yoy_earnings_growth_pct"), [(20, 20), (10, 15), (0, 10)], 0))
    add("revenue_growth", 10, band(f.get("yoy_revenue_growth_pct"), [(15, 10), (5, 7), (0, 4)], 0))
    add("net_margin", 10, band(f.get("net_margin_pct"), [(15, 10), (8, 7), (0, 4)], 0))
    roe = f.get("roe_ttm_pct") if f.get("roe_ttm_pct") is not None else f.get("roe_pct")
    add("roe", 15, band(roe, [(18, 15), (12, 11), (8, 7)], 2))

    pe = f.get("pe_ratio")
    if pe is None:
        add("valuation_pe", 15, None)
    elif pe <= 0:
        add("valuation_pe", 15, 0)
    else:
        add("valuation_pe", 15, 15 if pe < 15 else 11 if pe < 25 else 6 if pe < 40 else 2)

    if not is_financial:
        de = f.get("debt_to_equity")
        add("leverage", 10, None if de is None else 10 if de < 0.5 else 8 if de < 1 else 5 if de < 2 else 2)
        ic = f.get("interest_coverage")
        add("interest_coverage", 5, None if ic is None else 5 if ic > 3 else 2 if ic > 1.5 else 0)
        pf = f.get("piotroski_f_score")
        add("piotroski", 10, None if pf is None else 10 if pf >= 7 else 6 if pf >= 5 else 2)
    fcf = f.get("free_cash_flow_ttm_cr")
    add("free_cash_flow", 5, None if fcf is None else 5 if fcf > 0 else 0)

    max_total = sum(m for _, m, _ in items)
    earned = sum((p if p is not None else m / 2) for _, m, p in items)
    with_data = sum(1 for _, _, p in items if p is not None)
    return {
        "score": round(earned / max_total * 100, 1),
        "rating": rating_label(earned / max_total * 100),
        "data_coverage_pct": round(with_data / len(items) * 100, 1),
        "breakdown": [{"factor": n, "points": p, "max": m, "has_data": p is not None} for n, m, p in items],
        "note": "Leverage/interest-coverage/Piotroski skipped for financial companies." if is_financial else None,
    }


def combine_scores(tech_score, fund_score):
    """60% technical + 40% fundamental, only when fundamental coverage is decent."""
    if fund_score and fund_score["data_coverage_pct"] >= 50:
        overall = round(0.6 * tech_score["score"] + 0.4 * fund_score["score"], 1)
        basis = "60% technical + 40% fundamental"
    else:
        overall = float(tech_score["score"])
        basis = "technical only (insufficient fundamental data)"
    return {"score": overall, "rating": rating_label(overall), "basis": basis}


# ── RISK + RELATIVE STRENGTH ──────────────────────────────────────────────

def compute_risk_metrics(close, bench_close=None):
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        return {"error": "Not enough history for risk metrics"}

    mean_ann = returns.mean() * TRADING_DAYS
    vol_ann = returns.std() * math.sqrt(TRADING_DAYS)
    downside = returns[returns < 0].std() * math.sqrt(TRADING_DAYS)
    drawdown = close / close.cummax() - 1
    var95 = float(np.percentile(returns, 5))
    tail = returns[returns <= var95]
    years = len(returns) / TRADING_DAYS
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1 if years > 0 else None

    risk = {
        "period_start": str(close.index[0].date()),
        "period_end": str(close.index[-1].date()),
        "cagr_pct": safe_round(cagr * 100) if cagr is not None else None,
        "annualized_volatility_pct": safe_round(vol_ann * 100),
        "sharpe_ratio": safe_round((mean_ann - RISK_FREE_RATE) / vol_ann) if vol_ann else None,
        "sortino_ratio": safe_round((mean_ann - RISK_FREE_RATE) / downside) if downside and not pd.isna(downside) else None,
        "max_drawdown_pct": safe_round(drawdown.min() * 100),
        "current_drawdown_pct": safe_round(drawdown.iloc[-1] * 100),
        "var_95_daily_pct": safe_round(var95 * 100),
        "cvar_95_daily_pct": safe_round(tail.mean() * 100) if len(tail) else None,
        "risk_free_rate_assumed_pct": RISK_FREE_RATE * 100,
    }

    if bench_close is not None:
        joined = pd.concat([returns, bench_close.pct_change()], axis=1, join="inner").dropna()
        joined.columns = ["s", "b"]
        if len(joined) >= 30 and joined["b"].var() > 0:
            beta = joined["s"].cov(joined["b"]) / joined["b"].var()
            rf_d = RISK_FREE_RATE / TRADING_DAYS
            alpha_ann = ((joined["s"].mean() - rf_d) - beta * (joined["b"].mean() - rf_d)) * TRADING_DAYS
            corr = joined["s"].corr(joined["b"])
            risk.update({
                "beta": safe_round(beta),
                "alpha_annualized_pct": safe_round(alpha_ann * 100),
                "correlation_with_benchmark": safe_round(corr),
                "r_squared": safe_round(corr ** 2),
            })
    return risk


def compute_relative_strength(close, bench_close):
    df = pd.concat([close, bench_close], axis=1, join="inner").dropna()
    df.columns = ["s", "b"]
    out = {"benchmark": BENCHMARK, "periods": {}}
    for label, days in {"1m": 21, "3m": 63, "6m": 126, "1y": 252}.items():
        if len(df) > days:
            s_ret = df["s"].iloc[-1] / df["s"].iloc[-1 - days] - 1
            b_ret = df["b"].iloc[-1] / df["b"].iloc[-1 - days] - 1
            out["periods"][label] = {
                "stock_return_pct": safe_round(s_ret * 100),
                "benchmark_return_pct": safe_round(b_ret * 100),
                "excess_return_pct": safe_round((s_ret - b_ret) * 100),
                "outperforming": bool(s_ret > b_ret),
            }
    if len(df) >= 60:
        ratio = df["s"] / df["b"]
        out["rs_ratio_trend"] = "improving" if ratio.iloc[-1] > ratio.rolling(50).mean().iloc[-1] else "weakening"
    return out


# ── FUNDAMENTALS ──────────────────────────────────────────────────────────

def _row(df, labels):
    """First matching row of a statement as a newest-first numeric Series, or None."""
    if df is None or getattr(df, "empty", True):
        return None
    for label in ([labels] if isinstance(labels, str) else labels):
        if label in df.index:
            s = pd.to_numeric(df.loc[label], errors="coerce").dropna()
            if len(s):
                return s.sort_index(ascending=False)
    return None


def _ttm(series):
    """Sum of the latest 4 quarters, or None if fewer than 4 are available."""
    if series is None or len(series) < 4:
        return None
    return float(series.iloc[:4].sum())


def _pair(df, labels):
    """(latest, previous) annual values for a statement row."""
    s = _row(df, labels)
    if s is None:
        return None, None
    return float(s.iloc[0]), (float(s.iloc[1]) if len(s) > 1 else None)


def _ratio(a, b):
    return None if (a is None or b in (None, 0)) else a / b


def compute_piotroski(stock):
    """Piotroski F-Score (0-9) from annual statements. Checks that cannot be
    computed are skipped and reported via `checks_available`.
    Not meaningful for banks/NBFCs."""
    try:
        inc, bs, cf = stock.income_stmt, stock.balance_sheet, stock.cashflow
    except Exception:
        return None, 0, {}

    ni_c, ni_p = _pair(inc, ["Net Income"])
    rev_c, rev_p = _pair(inc, ["Total Revenue"])
    gp_c, gp_p = _pair(inc, ["Gross Profit"])
    ta_c, ta_p = _pair(bs, ["Total Assets"])
    cfo_c, _ = _pair(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    ca_c, ca_p = _pair(bs, ["Current Assets"])
    cl_c, cl_p = _pair(bs, ["Current Liabilities"])
    debt_c, debt_p = _pair(bs, ["Long Term Debt", "Total Debt"])
    sh_c, sh_p = _pair(bs, ["Ordinary Shares Number", "Share Issued"])

    roa_c, roa_p = _ratio(ni_c, ta_c), _ratio(ni_p, ta_p)
    lev_c, lev_p = _ratio(debt_c, ta_c), _ratio(debt_p, ta_p)
    cr_c, cr_p = _ratio(ca_c, cl_c), _ratio(ca_p, cl_p)
    gm_c, gm_p = _ratio(gp_c, rev_c), _ratio(gp_p, rev_p)
    at_c, at_p = _ratio(rev_c, ta_c), _ratio(rev_p, ta_p)

    def cmp(a, b, op):
        return None if (a is None or b is None) else bool(op(a, b))

    checks = {
        "roa_positive": None if roa_c is None else roa_c > 0,
        "operating_cash_flow_positive": None if cfo_c is None else cfo_c > 0,
        "roa_improving": cmp(roa_c, roa_p, lambda a, b: a > b),
        "cash_flow_exceeds_net_income": cmp(cfo_c, ni_c, lambda a, b: a > b),
        "leverage_decreasing": cmp(lev_c, lev_p, lambda a, b: a < b),
        "current_ratio_improving": cmp(cr_c, cr_p, lambda a, b: a > b),
        "no_share_dilution": cmp(sh_c, sh_p, lambda a, b: a <= b),
        "gross_margin_improving": cmp(gm_c, gm_p, lambda a, b: a > b),
        "asset_turnover_improving": cmp(at_c, at_p, lambda a, b: a > b),
    }
    available = [v for v in checks.values() if v is not None]
    if len(available) < 5:  # too little data to be meaningful
        return None, len(available), checks
    return int(sum(available)), len(available), checks


def get_fundamentals(stock):
    """Extended fundamentals: growth, margins, ROE, leverage, valuation, cash flow,
    quarterly trend and Piotroski F-Score.

    IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits
    that specific endpoint for many cloud hosting providers (Render, Railway, Heroku),
    returning an almost-empty dict even though the price-history endpoint works fine.
    Everything is calculated from the income statement, balance sheet and cash flow
    statement instead. Money values suffixed _cr are in Indian crore (1 cr = 1e7 INR)."""
    fundamentals = {
        # original keys
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
        # new keys
        "ttm_revenue_cr": None,
        "ttm_net_income_cr": None,
        "roe_ttm_pct": None,
        "pe_ratio": None,
        "pb_ratio": None,
        "ev_to_ebitda": None,
        "dividend_yield_pct": None,
        "interest_coverage": None,
        "current_ratio": None,
        "free_cash_flow_ttm_cr": None,
        "piotroski_f_score": None,
        "piotroski_checks_available": 0,
        "piotroski_checks": {},
        "quarterly_trend": [],
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None
    ttm_net_income = None
    equity = None
    total_debt = None
    cash = None
    income_stmt = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

        # ── new: TTM figures and quarterly trend ──
        ni_s = _row(income_stmt, "Net Income")
        rev_s = _row(income_stmt, "Total Revenue")
        ttm_net_income = _ttm(ni_s)
        ttm_rev = _ttm(rev_s)
        if ttm_net_income is not None:
            fundamentals["ttm_net_income_cr"] = safe_round(ttm_net_income / 1e7)
        if ttm_rev is not None:
            fundamentals["ttm_revenue_cr"] = safe_round(ttm_rev / 1e7)

        if rev_s is not None:
            trend = []
            for dt in list(rev_s.index)[:8]:
                rev_v = float(rev_s.loc[dt])
                ni_v = float(ni_s.loc[dt]) if (ni_s is not None and dt in ni_s.index) else None
                trend.append({
                    "quarter": str(pd.Timestamp(dt).date()),
                    "revenue_cr": safe_round(rev_v / 1e7),
                    "net_income_cr": safe_round(ni_v / 1e7) if ni_v is not None else None,
                    "net_margin_pct": safe_round(ni_v / rev_v * 100) if (ni_v is not None and rev_v) else None,
                })
            fundamentals["quarterly_trend"] = trend

        # interest coverage = TTM operating income / TTM interest expense
        ebit_ttm = _ttm(_row(income_stmt, ["Operating Income", "EBIT"]))
        interest_ttm = _ttm(_row(income_stmt, ["Interest Expense", "Interest Expense Non Operating"]))
        if ebit_ttm is not None and interest_ttm not in (None, 0):
            fundamentals["interest_coverage"] = safe_round(ebit_ttm / abs(interest_ttm))

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # original ROE: latest quarter annualized (x4); kept for backward compatibility
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)
                # new: proper trailing-twelve-month ROE
                if ttm_net_income is not None:
                    fundamentals["roe_ttm_pct"] = safe_round(ttm_net_income / equity * 100)

            ca = _row(balance_sheet, "Current Assets")
            cl = _row(balance_sheet, "Current Liabilities")
            if ca is not None and cl is not None and float(cl.iloc[0]) != 0:
                fundamentals["current_ratio"] = safe_round(float(ca.iloc[0]) / float(cl.iloc[0]))

            cash_s = _row(balance_sheet, ["Cash And Cash Equivalents",
                                          "Cash Cash Equivalents And Short Term Investments"])
            if cash_s is not None:
                cash = float(cash_s.iloc[0])

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # free cash flow (TTM)
    try:
        fcf_s = _row(stock.quarterly_cashflow, "Free Cash Flow")
        fcf_ttm = _ttm(fcf_s)
        if fcf_ttm is not None:
            fundamentals["free_cash_flow_ttm_cr"] = safe_round(fcf_ttm / 1e7)
    except Exception as e:
        fundamentals["_debug_cashflow_error"] = str(e)

    # market cap + valuation ratios. fast_info is lighter than .info and less likely
    # to be blocked; separate try so a failure here doesn't affect anything above
    market_cap = None
    last_price = None
    try:
        fast_info = stock.fast_info
        if fast_info:
            if fast_info.get("marketCap"):
                market_cap = float(fast_info["marketCap"])
                fundamentals["market_cap_cr"] = safe_round(market_cap / 1e7)
            if fast_info.get("lastPrice"):
                last_price = float(fast_info["lastPrice"])
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    try:
        if market_cap:
            if ttm_net_income and ttm_net_income > 0:
                fundamentals["pe_ratio"] = safe_round(market_cap / ttm_net_income)
            if equity and equity > 0:
                fundamentals["pb_ratio"] = safe_round(market_cap / float(equity))
            ebitda_ttm = _ttm(_row(income_stmt, ["EBITDA", "Normalized EBITDA"]))
            if ebitda_ttm and ebitda_ttm > 0 and total_debt is not None:
                ev = market_cap + float(total_debt) - (cash or 0)
                fundamentals["ev_to_ebitda"] = safe_round(ev / ebitda_ttm)
        if last_price:
            div = stock.dividends
            if div is not None and len(div):
                tz = getattr(div.index, "tz", None)
                cutoff = pd.Timestamp.now(tz=tz) - pd.Timedelta(days=365)
                paid = float(div[div.index >= cutoff].sum())
                fundamentals["dividend_yield_pct"] = safe_round(paid / last_price * 100)
    except Exception as e:
        fundamentals["_debug_valuation_error"] = str(e)

    # Piotroski F-Score (annual statements)
    try:
        score, available, checks = compute_piotroski(stock)
        fundamentals["piotroski_f_score"] = score
        fundamentals["piotroski_checks_available"] = available
        fundamentals["piotroski_checks"] = checks
    except Exception as e:
        fundamentals["_debug_piotroski_error"] = str(e)

    return fundamentals


def get_fundamentals_cached(ticker: str):
    key = ("fund", ticker)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    result = get_fundamentals(yf.Ticker(ticker))
    _cache.set(key, result, FUNDAMENTALS_TTL)
    return result


def get_company_info(ticker: str):
    """Name + sector. Uses the built-in universe first, then a guarded Yahoo lookup
    (which may be blocked on cloud hosts), then falls back to the ticker."""
    base = base_symbol(ticker)
    if base in UNIVERSE:
        return UNIVERSE[base]
    key = ("info", ticker)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    name, sector = ticker, None
    try:
        info = yf.Ticker(ticker).get_info() or {}
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector")
    except Exception:
        pass
    _cache.set(key, (name, sector), FUNDAMENTALS_TTL)
    return name, sector


def is_financial_sector(sector):
    return bool(sector) and "financ" in sector.lower()


# ── SHARED ANALYSIS BUILDER ───────────────────────────────────────────────

def build_analysis(ticker: str, include_fundamentals: bool = True, include_risk: bool = True):
    hist = require_history(ticker)
    technical, corporate_action_flag = get_technical_snapshot(hist)
    levels, patterns = get_levels_and_patterns(hist)
    name, sector = get_company_info(ticker)

    tech_score = compute_signal_score(technical)
    result = {
        "symbol": ticker,
        "company_name": name,
        "sector": sector,
        "timestamp": datetime.now().isoformat(),
        "corporate_action_flag": corporate_action_flag,
        "corporate_action_note": (
            "A >15% single-day price move was detected in the last 2 years — "
            "moving-average and Bollinger figures may be distorted by a demerger, "
            "split, or similar event. Verify before trusting the technical signals."
            if corporate_action_flag else None
        ),
        "technical": technical,
        "levels": levels,
        "candlestick_patterns": patterns,
        "technical_score": tech_score,
    }

    fund_score = None
    if include_fundamentals:
        fundamentals = get_fundamentals_cached(ticker)
        fund_score = compute_fundamental_score(fundamentals, is_financial_sector(sector))
        result["fundamentals"] = fundamentals
        result["fundamental_score"] = fund_score
    result["overall"] = combine_scores(tech_score, fund_score)

    if include_risk:
        bench = get_benchmark_close()
        result["risk"] = compute_risk_metrics(hist["Close"], bench)
        if bench is not None:
            result["relative_strength"] = compute_relative_strength(hist["Close"], bench)
    return result


# ── ORIGINAL ENDPOINTS (enriched) ─────────────────────────────────────────

@app.get("/", tags=["Core"])
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API is running", "version": "2.0.0",
            "docs": "/docs", "statsmodels_available": HAS_STATSMODELS}


@app.get("/analyze/{symbol}", tags=["Core"])
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company, with
    signal scores, support/resistance, candlestick patterns, risk and relative strength."""
    ticker = to_ticker(symbol)
    try:
        return sanitize(build_analysis(ticker))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}", tags=["Core"])
def chart_data(symbol: str, months: int = 6):
    """
    OHLC candlestick data + overlay indicator series for charting.
    One row per trading day: date, OHLCV, RSI, MACD line/signal/histogram, MA50, MA200,
    Bollinger upper/lower, 20-day VWAP and OBV.
    `months` controls how much history to return in the chart (default 6, max 24).
    """
    ticker = to_ticker(symbol)
    months = min(max(months, 1), 24)

    try:
        hist = require_history(ticker)  # 2y so MA200 has enough lookback

        close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, macd_hist = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        vwap = calculate_vwap(high, low, close, volume)
        obv = calculate_obv(close, volume)

        recent_idx = hist.tail(months * 21).index  # ~21 trading days per month
        rows = []
        for date in recent_idx:
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(hist.at[date, "Open"]),
                "high": safe_round(hist.at[date, "High"]),
                "low": safe_round(hist.at[date, "Low"]),
                "close": safe_round(hist.at[date, "Close"]),
                "volume": int(hist.at[date, "Volume"]) if not pd.isna(hist.at[date, "Volume"]) else None,
                "rsi": safe_round(rsi.at[date]),
                "macd_line": safe_round(macd_line.at[date]),
                "macd_signal": safe_round(signal_line.at[date]),
                "macd_histogram": safe_round(macd_hist.at[date]),
                "ma_50": safe_round(ma50.at[date]),
                "ma_200": safe_round(ma200.at[date]),
                "bollinger_upper": safe_round(mid.at[date] + 2 * std.at[date]),
                "bollinger_lower": safe_round(mid.at[date] - 2 * std.at[date]),
                "vwap_20d": safe_round(vwap.at[date]),
                "obv": safe_round(obv.at[date], 0),
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist", tags=["Core"])
def watchlist(symbols: str, sort_by: Optional[str] = Query(
        None, description="score | rsi | change | volume_ratio | pct_of_52w_high")):
    """
    Multi-stock snapshot for a watchlist view, fetched in parallel.
    Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY
    Capped at 25 symbols per request. Optional sort_by (descending).
    """
    symbol_list = parse_symbols(symbols, MAX_WATCHLIST)
    tickers = [to_ticker(s) for s in symbol_list]
    histories = get_histories(tickers)

    results = []
    for ticker in tickers:
        hist = histories.get(ticker)
        if hist is None:
            results.append({"symbol": ticker, "error": "No data found"})
            continue
        try:
            technical, corporate_action_flag = get_technical_snapshot(hist)
            score = compute_signal_score(technical)
            name, sector = get_company_info(ticker)
            results.append({
                "symbol": ticker,
                "company_name": name,
                "sector": sector,
                "current_price": technical["current_price"],
                "day_change_pct": technical["day_change_pct"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "volume_ratio_20d": technical["volume_ratio_20d"],
                "technical_score": score["score"],
                "rating": score["rating"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    sort_keys = {"score": "technical_score", "rsi": "rsi_14", "change": "day_change_pct",
                 "volume_ratio": "volume_ratio_20d", "pct_of_52w_high": "pct_of_52w_high"}
    if sort_by in sort_keys:
        k = sort_keys[sort_by]
        results.sort(key=lambda r: (r.get(k) is None, -(r.get(k) or 0)))

    return sanitize({"count": len(results), "stocks": results})


# ── NEW: RISK, COMPARE, CORRELATION, SECTORS ──────────────────────────────

@app.get("/risk/{symbol}", tags=["Analysis"])
def risk(symbol: str, benchmark: str = BENCHMARK):
    """Beta, alpha, Sharpe, Sortino, max drawdown, VaR/CVaR and relative strength vs a benchmark."""
    ticker = to_ticker(symbol)
    hist = require_history(ticker)
    bench = get_benchmark_close(benchmark=benchmark)
    out = {"symbol": ticker, "benchmark": benchmark, "risk": compute_risk_metrics(hist["Close"], bench)}
    if bench is not None:
        out["relative_strength"] = compute_relative_strength(hist["Close"], bench)
    return sanitize(out)


@app.get("/compare", tags=["Analysis"])
def compare(symbols: str):
    """Peer comparison for 2-6 stocks: key metrics side by side, ranked by overall score.
    Example: /compare?symbols=HDFCBANK,ICICIBANK,SBIN"""
    symbol_list = parse_symbols(symbols, 6, minimum=2)
    tickers = [to_ticker(s) for s in symbol_list]

    def one(t):
        try:
            return build_analysis(t, include_fundamentals=True, include_risk=True)
        except Exception as e:
            return {"symbol": t, "error": str(getattr(e, "detail", e))}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as pool:
        analyses = list(pool.map(one, tickers))

    rows = []
    for a in analyses:
        if "error" in a:
            rows.append(a)
            continue
        t, f, r = a["technical"], a.get("fundamentals", {}), a.get("risk", {})
        rows.append({
            "symbol": a["symbol"], "company_name": a["company_name"], "sector": a["sector"],
            "price": t["current_price"], "overall_score": a["overall"]["score"], "rating": a["overall"]["rating"],
            "technical_score": a["technical_score"]["score"],
            "fundamental_score": a.get("fundamental_score", {}).get("score"),
            "rsi_14": t["rsi_14"], "ma_trend": t["ma_trend"], "pct_of_52w_high": t["pct_of_52w_high"],
            "pe_ratio": f.get("pe_ratio"), "pb_ratio": f.get("pb_ratio"), "roe_ttm_pct": f.get("roe_ttm_pct"),
            "yoy_earnings_growth_pct": f.get("yoy_earnings_growth_pct"),
            "net_margin_pct": f.get("net_margin_pct"), "debt_to_equity": f.get("debt_to_equity"),
            "dividend_yield_pct": f.get("dividend_yield_pct"),
            "beta": r.get("beta"), "volatility_pct": r.get("annualized_volatility_pct"),
            "sharpe_ratio": r.get("sharpe_ratio"), "max_drawdown_pct": r.get("max_drawdown_pct"),
        })

    valid = [r for r in rows if "error" not in r]

    def best(key, reverse):
        cands = [r for r in valid if r.get(key) is not None and (key != "pe_ratio" or r[key] > 0)]
        if not cands:
            return None
        return sorted(cands, key=lambda r: r[key], reverse=reverse)[0]["symbol"]

    return sanitize({
        "ranking": [r["symbol"] for r in sorted(valid, key=lambda r: -r["overall_score"])],
        "best_by": {
            "overall_score": best("overall_score", True),
            "lowest_pe": best("pe_ratio", False),
            "highest_roe": best("roe_ttm_pct", True),
            "highest_earnings_growth": best("yoy_earnings_growth_pct", True),
            "lowest_volatility": best("volatility_pct", False),
            "best_sharpe": best("sharpe_ratio", True),
        },
        "stocks": rows,
    })


@app.get("/correlation", tags=["Analysis"])
def correlation(symbols: str, days: int = Query(252, ge=60, le=500)):
    """Correlation matrix of daily returns for 2-15 stocks (diversification check)."""
    symbol_list = parse_symbols(symbols, 15, minimum=2)
    tickers = [to_ticker(s) for s in symbol_list]
    histories = get_histories(tickers)
    closes = {t: h["Close"] for t, h in histories.items() if h is not None}
    missing = [t for t in tickers if t not in closes]
    if len(closes) < 2:
        raise HTTPException(status_code=404, detail="Need data for at least two symbols.")

    rets = pd.DataFrame(closes).pct_change().dropna().tail(days)
    corr = rets.corr()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"a": cols[i], "b": cols[j], "correlation": safe_round(corr.iloc[i, j], 3)})
    pairs.sort(key=lambda p: p["correlation"] if p["correlation"] is not None else 0)
    return sanitize({
        "observations": len(rets), "missing_symbols": missing,
        "matrix": corr.round(3).to_dict(),
        "most_diversifying_pairs": pairs[:3],
        "most_correlated_pairs": list(reversed(pairs[-3:])),
        "average_pairwise_correlation": safe_round(np.mean([p["correlation"] for p in pairs]), 3) if pairs else None,
    })


@app.get("/sectors", tags=["Analysis"])
def sector_heatmap():
    """Average 1d / 1w / 1m / 3m return by sector across the built-in universe."""
    tickers = [to_ticker(s) for s in UNIVERSE]
    histories = get_histories(tickers)

    def ret(close, days):
        return (close.iloc[-1] / close.iloc[-1 - days] - 1) * 100 if len(close) > days else np.nan

    stock_rows = []
    for t, h in histories.items():
        if h is None:
            continue
        c = h["Close"]
        name, sector = get_company_info(t)
        stock_rows.append({"symbol": t, "name": name, "sector": sector,
                           "1d": ret(c, 1), "1w": ret(c, 5), "1m": ret(c, 21), "3m": ret(c, 63)})
    if not stock_rows:
        raise HTTPException(status_code=502, detail="Could not fetch any price data.")

    df = pd.DataFrame(stock_rows)
    sectors = []
    for sector, g in df.groupby("sector"):
        sectors.append({
            "sector": sector, "stock_count": int(len(g)),
            "avg_return_pct": {k: safe_round(g[k].mean()) for k in ["1d", "1w", "1m", "3m"]},
            "best_stock_1m": g.sort_values("1m", ascending=False).iloc[0]["symbol"],
            "worst_stock_1m": g.sort_values("1m").iloc[0]["symbol"],
        })
    sectors.sort(key=lambda s: (s["avg_return_pct"]["1m"] is None, -(s["avg_return_pct"]["1m"] or 0)))
    return sanitize({"as_of": datetime.now().isoformat(), "sectors": sectors, "stocks": stock_rows})


# ── NEW: SCREENER ─────────────────────────────────────────────────────────

@app.get("/screen", tags=["Screener"])
def screen(
    universe: str = Query("nifty50", description="'nifty50' (built-in list) or use `symbols`"),
    symbols: Optional[str] = Query(None, description="Comma-separated custom universe (max 60)"),
    # technical filters
    min_rsi: Optional[float] = None, max_rsi: Optional[float] = None,
    min_score: Optional[float] = Query(None, description="Minimum technical score (0-100)"),
    min_adx: Optional[float] = None,
    min_pct_52w_high: Optional[float] = None,
    min_volume_ratio: Optional[float] = None,
    above_ma200: Optional[bool] = None,
    ma_trend: Optional[str] = Query(None, description="bullish | bearish"),
    # fundamental filters (these trigger statement downloads: slower on first call)
    min_roe: Optional[float] = None, max_de: Optional[float] = None, max_pe: Optional[float] = None,
    min_earnings_growth: Optional[float] = None, min_piotroski: Optional[int] = None,
    limit: int = Query(20, ge=1, le=60),
):
    """Filter a universe on technical and fundamental rules, ranked by technical score.
    Example: /screen?max_rsi=35&above_ma200=true&min_roe=15&max_de=1
    Technical filters run first; fundamentals are only fetched for the survivors."""
    if symbols:
        symbol_list = parse_symbols(symbols, MAX_UNIVERSE)
    elif universe.lower() == "nifty50":
        symbol_list = list(UNIVERSE.keys())
    else:
        raise HTTPException(status_code=400, detail="universe must be 'nifty50', or pass `symbols`.")

    tickers = [to_ticker(s) for s in symbol_list]
    histories = get_histories(tickers)

    survivors = []
    for t in tickers:
        h = histories.get(t)
        if h is None:
            continue
        try:
            tech, _ = get_technical_snapshot(h)
            score = compute_signal_score(tech)
        except Exception:
            continue
        price, ma200 = tech["current_price"], tech["ma_200"]
        checks = [
            min_rsi is None or (tech["rsi_14"] is not None and tech["rsi_14"] >= min_rsi),
            max_rsi is None or (tech["rsi_14"] is not None and tech["rsi_14"] <= max_rsi),
            min_score is None or score["score"] >= min_score,
            min_adx is None or (tech["adx_14"] is not None and tech["adx_14"] >= min_adx),
            min_pct_52w_high is None or (tech["pct_of_52w_high"] is not None and tech["pct_of_52w_high"] >= min_pct_52w_high),
            min_volume_ratio is None or (tech["volume_ratio_20d"] is not None and tech["volume_ratio_20d"] >= min_volume_ratio),
            above_ma200 is None or (ma200 is not None and ((price > ma200) == above_ma200)),
            ma_trend is None or tech["ma_trend"] == ma_trend.lower(),
        ]
        if all(checks):
            survivors.append({"ticker": t, "tech": tech, "score": score})

    need_fundamentals = any(v is not None for v in [min_roe, max_de, max_pe, min_earnings_growth, min_piotroski])
    fundamentals_map = {}
    if need_fundamentals and survivors:
        def fetch(t):
            try:
                return t, get_fundamentals_cached(t)
            except Exception:
                return t, None
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            fundamentals_map = dict(pool.map(fetch, [s["ticker"] for s in survivors]))

    matches = []
    for s in survivors:
        entry = {
            "symbol": s["ticker"], "company_name": get_company_info(s["ticker"])[0],
            "sector": get_company_info(s["ticker"])[1],
            "price": s["tech"]["current_price"], "rsi_14": s["tech"]["rsi_14"],
            "ma_trend": s["tech"]["ma_trend"], "adx_14": s["tech"]["adx_14"],
            "pct_of_52w_high": s["tech"]["pct_of_52w_high"], "volume_ratio_20d": s["tech"]["volume_ratio_20d"],
            "technical_score": s["score"]["score"], "rating": s["score"]["rating"],
        }
        if need_fundamentals:
            f = fundamentals_map.get(s["ticker"])
            if not f:
                continue  # can't verify fundamental filters -> exclude
            roe = f.get("roe_ttm_pct") if f.get("roe_ttm_pct") is not None else f.get("roe_pct")
            fund_checks = [
                min_roe is None or (roe is not None and roe >= min_roe),
                max_de is None or (f.get("debt_to_equity") is not None and f["debt_to_equity"] <= max_de),
                max_pe is None or (f.get("pe_ratio") is not None and 0 < f["pe_ratio"] <= max_pe),
                min_earnings_growth is None or (f.get("yoy_earnings_growth_pct") is not None
                                                and f["yoy_earnings_growth_pct"] >= min_earnings_growth),
                min_piotroski is None or (f.get("piotroski_f_score") is not None and f["piotroski_f_score"] >= min_piotroski),
            ]
            if not all(fund_checks):
                continue
            entry.update({"roe_pct": roe, "debt_to_equity": f.get("debt_to_equity"), "pe_ratio": f.get("pe_ratio"),
                          "yoy_earnings_growth_pct": f.get("yoy_earnings_growth_pct"),
                          "piotroski_f_score": f.get("piotroski_f_score")})
        matches.append(entry)

    matches.sort(key=lambda m: -m["technical_score"])
    return sanitize({"universe_size": len(tickers), "data_available": sum(1 for t in tickers if histories.get(t) is not None),
                     "match_count": len(matches), "matches": matches[:limit]})


# ── NEW: ALERTS (n8n-friendly) ────────────────────────────────────────────

def _crossed_above(a, b, bars):
    """True if series `a` crossed from <= b to > b within the last `bars` bars."""
    return bool(((a.shift(1) <= b.shift(1)) & (a > b)).tail(bars).any())


def _crossed_below(a, b, bars):
    return bool(((a.shift(1) >= b.shift(1)) & (a < b)).tail(bars).any())


def compute_alerts(hist, lookback=1):
    close, volume = hist["Close"], hist["Volume"]
    rsi = calculate_rsi(close)
    macd_line, signal_line, _ = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    percent_b, _ = calculate_bollinger_bands(close)
    vol_avg = volume.rolling(20).mean()

    hi70 = pd.Series(70.0, index=rsi.index)
    lo30 = pd.Series(30.0, index=rsi.index)
    prior_high = close.rolling(252).max().shift(1)
    prior_low = close.rolling(252).min().shift(1)

    alerts = []

    def add(kind, severity, message):
        alerts.append({"type": kind, "severity": severity, "message": message})

    if _crossed_above(rsi, hi70, lookback):
        add("rsi_overbought", "warning", "RSI crossed above 70 (overbought).")
    if _crossed_below(rsi, lo30, lookback):
        add("rsi_oversold", "info", "RSI crossed below 30 (oversold).")
    if _crossed_above(macd_line, signal_line, lookback):
        add("macd_bullish_cross", "info", "MACD crossed above its signal line.")
    if _crossed_below(macd_line, signal_line, lookback):
        add("macd_bearish_cross", "warning", "MACD crossed below its signal line.")
    if _crossed_above(ma50, ma200, lookback):
        add("golden_cross", "info", "50DMA crossed above 200DMA (golden cross).")
    if _crossed_below(ma50, ma200, lookback):
        add("death_cross", "warning", "50DMA crossed below 200DMA (death cross).")
    if bool((close > prior_high).tail(lookback).any()):
        add("new_52w_high", "info", "Closed at a new 52-week high.")
    if bool((close < prior_low).tail(lookback).any()):
        add("new_52w_low", "warning", "Closed at a new 52-week low.")
    if not pd.isna(vol_avg.iloc[-1]) and vol_avg.iloc[-1] > 0 and volume.iloc[-1] >= 2 * vol_avg.iloc[-1]:
        add("volume_spike", "info", f"Volume is {volume.iloc[-1] / vol_avg.iloc[-1]:.1f}x the 20-day average.")
    if not pd.isna(percent_b.iloc[-1]):
        if percent_b.iloc[-1] > 1:
            add("bollinger_upper_break", "info", "Price closed above the upper Bollinger band.")
        elif percent_b.iloc[-1] < 0:
            add("bollinger_lower_break", "info", "Price closed below the lower Bollinger band.")
    return alerts


@app.get("/alerts", tags=["Alerts"])
def alerts(symbols: str, lookback: int = Query(1, ge=1, le=10,
          description="Look back this many trading days for crossovers (1 = today only)")):
    """Triggered technical alerts for a watchlist. Schedule this in n8n and branch on
    `any_triggered`; `alerts_flat` is a ready-to-send list for Telegram/email/Slack.
    Example: /alerts?symbols=RELIANCE,TCS,INFY&lookback=2"""
    symbol_list = parse_symbols(symbols, MAX_WATCHLIST)
    tickers = [to_ticker(s) for s in symbol_list]
    histories = get_histories(tickers)

    per_symbol, flat = [], []
    for t in tickers:
        h = histories.get(t)
        if h is None:
            per_symbol.append({"symbol": t, "error": "No data found"})
            continue
        try:
            found = compute_alerts(h, lookback)
        except Exception as e:
            per_symbol.append({"symbol": t, "error": str(e)})
            continue
        price = safe_round(h["Close"].iloc[-1])
        per_symbol.append({"symbol": t, "price": price, "alert_count": len(found), "alerts": found})
        for a in found:
            flat.append({"symbol": t, "price": price, **a})
    return sanitize({"checked_at": datetime.now().isoformat(), "lookback_days": lookback,
                     "any_triggered": bool(flat), "total_alerts": len(flat),
                     "alerts_flat": flat, "by_symbol": per_symbol})


# ── NEW: BACKTESTING ──────────────────────────────────────────────────────

def _hold_from_signals(entry, exit_):
    """Stateful long-only position: enter on `entry`, stay until `exit_`."""
    holding, out = 0, []
    for e, x in zip(entry.values, exit_.values):
        if holding == 0 and e:
            holding = 1
        elif holding == 1 and x:
            holding = 0
        out.append(holding)
    return pd.Series(out, index=entry.index)


def build_position(close, strategy, fast, slow, oversold, overbought):
    if strategy == "golden_cross":
        f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
        return (f > s).astype(int)
    if strategy == "macd":
        m, sig, _ = calculate_macd(close)
        return (m > sig).astype(int)
    if strategy == "rsi":
        rsi = calculate_rsi(close)
        return _hold_from_signals(rsi < oversold, rsi > overbought)
    if strategy == "bollinger":
        pb, _ = calculate_bollinger_bands(close)
        return _hold_from_signals(pb < 0, pb > 0.5)
    raise HTTPException(status_code=400, detail="strategy must be golden_cross, rsi, macd or bollinger")


def run_backtest(close, strategy, fast=50, slow=200, oversold=30, overbought=70, cost_pct=0.1):
    """Long-only, one position at a time. Signals are computed on day t's close and
    traded on day t+1 (no look-ahead). `cost_pct` is charged on every entry and exit."""
    signal = build_position(close, strategy, fast, slow, oversold, overbought)
    pos = signal.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    cost = pos.diff().abs().fillna(pos.abs()) * (cost_pct / 100)
    strat = pos * ret - cost
    equity = (1 + strat).cumprod()
    buy_hold = (1 + ret).cumprod()

    years = len(close) / TRADING_DAYS
    std = strat.std()
    sharpe = ((strat.mean() * TRADING_DAYS - RISK_FREE_RATE) / (std * math.sqrt(TRADING_DAYS))) if std and std > 0 else None

    df = pd.DataFrame({"pos": pos, "r": strat})
    df["tid"] = (df["pos"].diff().fillna(df["pos"]) == 1).cumsum()
    held = df[df["pos"] == 1]
    trades = []
    if len(held):
        grp = held.groupby("tid")
        tret = grp["r"].apply(lambda s: (1 + s).prod() - 1)
        span = held.index.to_series().groupby(held["tid"]).agg(["first", "last"])
        for tid in tret.index:
            trades.append({"entry": str(span.loc[tid, "first"].date()), "exit": str(span.loc[tid, "last"].date()),
                           "return_pct": safe_round(tret.loc[tid] * 100)})
    wins = [t for t in trades if t["return_pct"] is not None and t["return_pct"] > 0]

    def cagr(curve):
        return (curve.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 and curve.iloc[-1] > 0 else None

    return {
        "period_start": str(close.index[0].date()), "period_end": str(close.index[-1].date()),
        "strategy_total_return_pct": safe_round((equity.iloc[-1] - 1) * 100),
        "strategy_cagr_pct": safe_round(cagr(equity)),
        "buy_and_hold_return_pct": safe_round((buy_hold.iloc[-1] - 1) * 100),
        "buy_and_hold_cagr_pct": safe_round(cagr(buy_hold)),
        "strategy_max_drawdown_pct": safe_round((equity / equity.cummax() - 1).min() * 100),
        "buy_and_hold_max_drawdown_pct": safe_round((buy_hold / buy_hold.cummax() - 1).min() * 100),
        "sharpe_ratio": safe_round(sharpe),
        "time_in_market_pct": safe_round(pos.mean() * 100),
        "trade_count": len(trades),
        "win_rate_pct": safe_round(len(wins) / len(trades) * 100) if trades else None,
        "avg_trade_return_pct": safe_round(np.mean([t["return_pct"] for t in trades])) if trades else None,
        "last_trades": trades[-10:],
    }


@app.get("/backtest/{symbol}", tags=["Backtest"])
def backtest(
    symbol: str,
    strategy: str = Query("golden_cross", description="golden_cross | rsi | macd | bollinger"),
    years: int = Query(5, ge=1, le=10),
    fast: int = Query(50, ge=2, le=100), slow: int = Query(200, ge=10, le=400),
    oversold: float = 30, overbought: float = 70,
    cost_pct: float = Query(0.1, ge=0, le=2, description="Cost per entry/exit, in %"),
):
    """Simple long-only backtest versus buy-and-hold.
    golden_cross: hold while fast MA > slow MA · macd: hold while MACD > signal
    rsi: buy RSI<oversold, sell RSI>overbought · bollinger: buy %B<0, sell %B>0.5.
    Past performance does not predict future returns."""
    ticker = to_ticker(symbol)
    period = "1y" if years <= 1 else "2y" if years <= 2 else "5y" if years <= 5 else "10y"
    hist = require_history(ticker, period)
    if len(hist) < 260:
        raise HTTPException(status_code=400, detail="Not enough history to backtest.")
    result = run_backtest(hist["Close"], strategy, fast, slow, oversold, overbought, cost_pct)
    return sanitize({"symbol": ticker, "strategy": strategy, "period_requested": period,
                     "assumptions": {"signals_traded_next_day": True, "cost_pct_per_side": cost_pct,
                                     "long_only": True, "adjusted_prices": True}, **result})


# ── NEW: PORTFOLIO ────────────────────────────────────────────────────────

class Holding(BaseModel):
    symbol: str
    quantity: float = Field(gt=0)
    avg_price: float = Field(gt=0, description="Your average buy price per share (INR)")


class PortfolioRequest(BaseModel):
    holdings: List[Holding] = Field(min_length=1, max_length=40)


@app.post("/portfolio", tags=["Portfolio"])
def portfolio(req: PortfolioRequest):
    """P&L, weights, sector mix, concentration and portfolio-level risk (stateless: send your
    holdings in each request). Risk uses current weights applied to ~2 years of returns."""
    tickers = [to_ticker(h.symbol) for h in req.holdings]
    histories = get_histories(list(dict.fromkeys(tickers)))

    rows, closes = [], {}
    for h, t in zip(req.holdings, tickers):
        hist = histories.get(t)
        if hist is None:
            rows.append({"symbol": t, "error": "No data found"})
            continue
        price = float(hist["Close"].iloc[-1])
        invested = h.quantity * h.avg_price
        value = h.quantity * price
        name, sector = get_company_info(t)
        closes[t] = hist["Close"]
        rows.append({"symbol": t, "company_name": name, "sector": sector, "quantity": h.quantity,
                     "avg_price": h.avg_price, "current_price": safe_round(price),
                     "invested": safe_round(invested), "current_value": safe_round(value),
                     "pnl": safe_round(value - invested), "pnl_pct": safe_round((value / invested - 1) * 100),
                     "day_change_pct": safe_round(hist["Close"].pct_change().iloc[-1] * 100)})

    valid = [r for r in rows if "error" not in r]
    if not valid:
        raise HTTPException(status_code=404, detail="No price data for any holding.")

    total_value = sum(r["current_value"] for r in valid)
    total_invested = sum(r["invested"] for r in valid)
    for r in valid:
        r["weight_pct"] = safe_round(r["current_value"] / total_value * 100)

    sector_mix: Dict[str, float] = {}
    for r in valid:
        sector_mix[r["sector"] or "Unknown"] = sector_mix.get(r["sector"] or "Unknown", 0) + r["weight_pct"]

    weights = pd.Series({r["symbol"]: r["current_value"] / total_value for r in valid})
    weights = weights.groupby(level=0).sum()
    hhi = float((weights ** 2).sum())

    summary = {
        "total_invested": safe_round(total_invested), "total_value": safe_round(total_value),
        "total_pnl": safe_round(total_value - total_invested),
        "total_pnl_pct": safe_round((total_value / total_invested - 1) * 100),
        "holding_count": len(valid),
        "largest_position": max(valid, key=lambda r: r["weight_pct"])["symbol"],
        "largest_weight_pct": max(r["weight_pct"] for r in valid),
        "concentration_hhi": safe_round(hhi, 3),
        "effective_number_of_stocks": safe_round(1 / hhi) if hhi else None,
        "sector_weights_pct": {k: safe_round(v) for k, v in sorted(sector_mix.items(), key=lambda kv: -kv[1])},
    }

    risk_out = None
    if len(closes) >= 1:
        rets = pd.DataFrame(closes).pct_change().dropna()
        w = weights.reindex(rets.columns).fillna(0)
        w = w / w.sum()
        port_ret = (rets * w).sum(axis=1)
        port_curve = (1 + port_ret).cumprod()
        bench = get_benchmark_close()
        risk_out = compute_risk_metrics(port_curve, bench)
        if len(rets.columns) > 1:
            c = rets.corr().values
            n = len(c)
            risk_out["average_pairwise_correlation"] = safe_round((c.sum() - n) / (n * (n - 1)), 3)
        risk_out["note"] = "Based on current weights applied to ~2y of history (not your actual trade history)."

    return sanitize({"summary": summary, "holdings": rows, "risk": risk_out})


# ── NEW: QUANT TOOLS ──────────────────────────────────────────────────────

def hurst_exponent(log_prices, max_lag=100):
    """H<0.5 mean-reverting, H~0.5 random walk, H>0.5 trending."""
    x = np.asarray(log_prices, dtype=float)
    lags = range(2, min(max_lag, len(x) // 4))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    tau = np.array(tau)
    if len(tau) < 5 or (tau <= 0).any():
        return None
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])


def half_life(series):
    """Mean-reversion half-life in days from an AR(1) fit on the level series (Ornstein-Uhlenbeck)."""
    y = np.asarray(series, dtype=float)
    lagged, delta = y[:-1], np.diff(y)
    if len(lagged) < 30:
        return None
    b = np.polyfit(lagged, delta, 1)[0]
    phi = 1 + b  # AR(1) coefficient; mean-reverting only if 0 < phi < 1
    return float(-math.log(2) / math.log(phi)) if 0 < phi < 1 else None


def simple_acf(x, nlags=10):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom == 0:
        return []
    return [safe_round((x[k:] * x[:-k]).sum() / denom, 4) for k in range(1, nlags + 1)]


def ewma_volatility(returns, lam=0.94):
    """RiskMetrics EWMA volatility; returns (next-day vol %, annualized vol %)."""
    r = np.asarray(returns, dtype=float)
    var = np.var(r[:30]) if len(r) >= 30 else np.var(r)
    for x in r:
        var = lam * var + (1 - lam) * x * x
    daily = math.sqrt(var)
    return daily * 100, daily * math.sqrt(TRADING_DAYS) * 100


def adf_summary(series):
    if not HAS_STATSMODELS:
        return {"available": False, "note": "pip install statsmodels to enable the ADF test"}
    stat, pvalue, _, _, crit, _ = adfuller(np.asarray(series, dtype=float), autolag="AIC")
    return {"available": True, "statistic": safe_round(stat, 3), "p_value": safe_round(pvalue, 4),
            "critical_values": {k: safe_round(v, 3) for k, v in crit.items()},
            "stationary_at_5pct": bool(pvalue < 0.05)}


@app.get("/quant/{symbol}", tags=["Quant"])
def quant(symbol: str):
    """Statistical profile: ADF stationarity tests (log price and returns), Hurst exponent,
    mean-reversion half-life, return autocorrelation, EWMA volatility forecast.
    Indicative statistics for research, not trading advice."""
    ticker = to_ticker(symbol)
    hist = require_history(ticker)
    close = hist["Close"]
    logp = np.log(close)
    rets = logp.diff().dropna()

    h = hurst_exponent(logp.values)
    regime = None if h is None else "mean-reverting" if h < 0.45 else "trending" if h > 0.55 else "random-walk-like"
    next_vol, ann_vol = ewma_volatility(rets.values)
    acf_vals = simple_acf(rets.values, 10)
    conf = 1.96 / math.sqrt(len(rets))
    hl = half_life(logp.values)

    return sanitize({
        "symbol": ticker, "observations": len(rets),
        "adf_log_price": adf_summary(logp.values),
        "adf_returns": adf_summary(rets.values),
        "hurst_exponent": safe_round(h, 3), "hurst_regime": regime,
        "mean_reversion_half_life_days": safe_round(hl, 1),
        "half_life_note": None if hl else "No mean reversion detected in price level (typical for stock prices).",
        "return_acf_lags_1_to_10": acf_vals,
        "acf_95pct_band": safe_round(conf, 4),
        "significant_acf_lags": [i + 1 for i, v in enumerate(acf_vals) if v is not None and abs(v) > conf],
        "ewma_volatility": {"lambda": 0.94, "next_day_vol_pct": safe_round(next_vol),
                            "annualized_vol_pct": safe_round(ann_vol),
                            "realized_annualized_vol_pct": safe_round(rets.tail(TRADING_DAYS).std() * math.sqrt(TRADING_DAYS) * 100)},
        "interpretation_note": "Prices are usually non-stationary (ADF p>0.05) while returns are stationary; "
                               "that is expected and is why models are fitted on returns.",
    })


@app.get("/pairs", tags=["Quant"])
def pairs(a: str, b: str, days: int = Query(504, ge=120, le=1000)):
    """Pairs-trading statistics for two stocks: correlation, hedge ratio, Engle-Granger
    cointegration p-value, spread half-life and current spread z-score.
    Example: /pairs?a=HDFCBANK&b=ICICIBANK"""
    ta, tb = to_ticker(a), to_ticker(b)
    hists = get_histories([ta, tb])
    if hists[ta] is None or hists[tb] is None:
        raise HTTPException(status_code=404, detail="No data for one or both symbols.")

    df = pd.concat([np.log(hists[ta]["Close"]), np.log(hists[tb]["Close"])], axis=1, join="inner").dropna().tail(days)
    df.columns = ["a", "b"]
    if len(df) < 120:
        raise HTTPException(status_code=400, detail="Not enough overlapping history.")

    beta, alpha = np.polyfit(df["a"].values, df["b"].values, 1)  # b = alpha + beta * a
    spread = df["b"] - (alpha + beta * df["a"])
    z = (spread.iloc[-1] - spread.mean()) / spread.std()

    coint_p = None
    if HAS_STATSMODELS:
        try:
            coint_p = float(coint(df["b"].values, df["a"].values)[1])
        except Exception:
            pass

    hl = half_life(spread.values)
    return sanitize({
        "a": ta, "b": tb, "observations": len(df),
        "return_correlation": safe_round(df.diff().dropna()["a"].corr(df.diff().dropna()["b"]), 3),
        "hedge_ratio_beta": safe_round(beta, 4), "intercept": safe_round(alpha, 4),
        "cointegration_p_value": safe_round(coint_p, 4),
        "cointegrated_at_5pct": None if coint_p is None else bool(coint_p < 0.05),
        "spread_adf": adf_summary(spread.values),
        "spread_half_life_days": safe_round(hl, 1),
        "current_spread_zscore": safe_round(z),
        "reading": (
            f"Spread is stretched ({z:+.2f} sd): {tb} looks expensive relative to {ta}" if z > 2 else
            f"Spread is stretched ({z:+.2f} sd): {tb} looks cheap relative to {ta}" if z < -2 else
            "Spread is within normal range"),
        "statsmodels_available": HAS_STATSMODELS,
        "note": "Cointegration found in-sample does not guarantee it persists. Research use only.",
    })


# ── NEW: NSE-SPECIFIC DATA ────────────────────────────────────────────────
# NOTE: These call NSE's public website API, which is unofficial and can change
# without notice. NSE also blocks many cloud/datacenter IPs (Render, Railway, AWS),
# so these endpoints may work on your laptop but fail once deployed. Response
# field names below follow NSE's current format; if NSE changes them, only this
# section needs updating. Failures return HTTP 502 with a clear message.

NSE_BASE = "https://www.nse-india.com"
NSE_INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_BASE + "/",
        })
        self._warmed_at = 0.0
        self._lock = threading.Lock()

    def _warm(self):
        self.session.get(NSE_BASE, timeout=10)  # NSE requires cookies from the home page
        self._warmed_at = time.time()

    def get(self, path, params=None):
        cache_key = ("nse", path, tuple(sorted((params or {}).items())))
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            with self._lock:
                if time.time() - self._warmed_at > 240:
                    self._warm()
            resp = self.session.get(NSE_BASE + path, params=params, timeout=15)
            if resp.status_code in (401, 403):
                with self._lock:
                    self._warm()
                resp = self.session.get(NSE_BASE + path, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=(
                f"NSE request failed ({e}). NSE often blocks cloud IPs and may rate-limit; try again later or run locally."))
        _cache.set(cache_key, data, NSE_TTL)
        return data


nse = NSEClient()


def compute_max_pain(rows):
    """Strike at which total option-buyer payout is minimised (sellers' 'pain' point)."""
    strikes = sorted({r["strikePrice"] for r in rows})
    ce = {r["strikePrice"]: (r.get("CE") or {}).get("openInterest", 0) or 0 for r in rows}
    pe = {r["strikePrice"]: (r.get("PE") or {}).get("openInterest", 0) or 0 for r in rows}
    best, best_pain = None, None
    for k in strikes:
        pain = sum(ce[s] * max(0, k - s) + pe[s] * max(0, s - k) for s in strikes)
        if best_pain is None or pain < best_pain:
            best, best_pain = k, pain
    return best


def analyse_option_chain(payload, expiry=None):
    records = payload.get("records", {})
    expiries = records.get("expiryDates", [])
    chosen = expiry or (expiries[0] if expiries else None)
    rows = [r for r in records.get("data", []) if chosen is None or r.get("expiryDate") == chosen]
    if not rows:
        raise HTTPException(status_code=404, detail="No option chain rows found (check symbol/expiry).")

    def oi(r, side): return (r.get(side) or {}).get("openInterest", 0) or 0
    def chg(r, side): return (r.get(side) or {}).get("changeinOpenInterest", 0) or 0

    total_ce = sum(oi(r, "CE") for r in rows)
    total_pe = sum(oi(r, "PE") for r in rows)
    underlying = records.get("underlyingValue")
    atm = min(rows, key=lambda r: abs(r["strikePrice"] - underlying))["strikePrice"] if underlying else None

    top_ce = sorted(rows, key=lambda r: oi(r, "CE"), reverse=True)[:3]
    top_pe = sorted(rows, key=lambda r: oi(r, "PE"), reverse=True)[:3]
    pcr = total_pe / total_ce if total_ce else None

    return {
        "expiry": chosen, "available_expiries": expiries[:6], "underlying_value": underlying,
        "atm_strike": atm,
        "total_call_oi": total_ce, "total_put_oi": total_pe,
        "pcr_oi": safe_round(pcr, 3),
        "pcr_reading": None if pcr is None else "put-heavy (often read as supportive)" if pcr > 1.2
                       else "call-heavy (often read as capped upside)" if pcr < 0.8 else "balanced",
        "net_change_in_call_oi": sum(chg(r, "CE") for r in rows),
        "net_change_in_put_oi": sum(chg(r, "PE") for r in rows),
        "max_pain_strike": compute_max_pain(rows),
        "resistance_strikes_by_call_oi": [{"strike": r["strikePrice"], "oi": oi(r, "CE")} for r in top_ce],
        "support_strikes_by_put_oi": [{"strike": r["strikePrice"], "oi": oi(r, "PE")} for r in top_pe],
    }


@app.get("/nse/option-chain/{symbol}", tags=["NSE data"])
def nse_option_chain(symbol: str, expiry: Optional[str] = Query(None, description="e.g. 28-Oct-2026; default nearest")):
    """PCR, max pain and OI-based support/resistance. Works for stocks and NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY."""
    sym = symbol.upper().strip()
    path = "/api/option-chain-indices" if sym in NSE_INDEX_SYMBOLS else "/api/option-chain-equities"
    payload = nse.get(path, {"symbol": sym})
    return sanitize({"symbol": sym, **analyse_option_chain(payload, expiry)})


@app.get("/nse/fii-dii", tags=["NSE data"])
def nse_fii_dii():
    """Latest FII/FPI and DII cash-market buy/sell/net values (INR crore)."""
    data = nse.get("/api/fiidiiTradeReact")
    return {"as_reported_by_nse": data}


@app.get("/nse/announcements/{symbol}", tags=["NSE data"])
def nse_announcements(symbol: str, limit: int = Query(15, ge=1, le=50)):
    """Recent corporate announcements (results, dividends, board meetings, etc.)."""
    sym = symbol.upper().strip()
    data = nse.get("/api/corporate-announcements", {"index": "equities", "symbol": sym})
    items = data if isinstance(data, list) else data.get("data", [])
    keep = ("an_dt", "desc", "attchmntText", "attchmntFile")
    return {"symbol": sym, "count": min(len(items), limit),
            "announcements": [{k: it.get(k) for k in keep} for it in items[:limit]]}


@app.get("/nse/bulk-deals", tags=["NSE data"])
def nse_bulk_deals(limit: int = Query(30, ge=1, le=100)):
    """Latest bulk, block and short-selling disclosures."""
    data = nse.get("/api/snapshot-capital-market-largedeal")
    return {k: (v[:limit] if isinstance(v, list) else v) for k, v in data.items()}


@app.get("/nse/delivery/{symbol}", tags=["NSE data"])
def nse_delivery(symbol: str):
    """Traded vs delivered quantity and delivery % (high delivery % = conviction buying/selling)."""
    sym = symbol.upper().strip()
    data = nse.get("/api/quote-equity", {"symbol": sym, "section": "trade_info"})
    dp = data.get("securityWiseDP") or {}
    if not dp:
        raise HTTPException(status_code=404, detail="Delivery data not available for this symbol.")
    return {"symbol": sym, "quantity_traded": dp.get("quantityTraded"),
            "delivery_quantity": dp.get("deliveryQuantity"),
            "delivery_pct": dp.get("deliveryToTradedQuantity"),
            "as_of": dp.get("secWiseDelPosDate")}


# ── UTILITY ───────────────────────────────────────────────────────────────

@app.post("/cache/clear", tags=["Utility"])
def clear_cache():
    """Drop all cached price history, statements and NSE responses."""
    return {"cleared_entries": _cache.clear()}
