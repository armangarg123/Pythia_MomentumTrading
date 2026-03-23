"""
pythia_dashboard.py
Pythia: Momentum Investing — Interactive Dashboard

Standalone Dash application.

Usage:
    uv run pythia_dashboard.py

Then open the URL shown in the terminal

Requirements:
    uv add dash dash-bootstrap-components plotly yfinance
    (duckdb pandas numpy scipy sklearn lightgbm already installed)

Tabs
    1  Strategy Backtester    single signal, custom parameters
    2  Stock Screener         live momentum ranking
    3  Signal Comparison      all signals side-by-side
    4  Fama-French            risk-adjust returns
    5  Machine Learning       ML vs Composite signal
"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================

import io
import warnings
import zipfile
from datetime import date
from functools import lru_cache
from pathlib import Path

import requests
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import duckdb
import yfinance as yf
import lightgbm as lgb

from scipy import stats
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from   plotly.subplots      import make_subplots

import dash
from   dash                      import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Paths — use Render's persistent disk (/data) when available, else local dir
import os as _os
_DATA_DIR = Path(_os.environ.get("DATA_DIR", "."))

DB_PATH         = _DATA_DIR / "sp500_prices.duckdb"
FF_CACHE        = _DATA_DIR / "ff3_factors.csv"
BENCHMARK_CACHE = _DATA_DIR / "benchmark_gspc.csv"

# Finance constants — identical to notebook Section 1
TRADING_DAYS         = 252
RF_ANNUAL            = 0.04
BACKTEST_START       = "2015-01-01"
TRANSACTION_COST_BPS = 10
TOP_DECILE           = 0.10
BOTTOM_DECILE        = 0.10
MIN_HISTORY_YEARS    = 8
MAX_DAILY_RETURN     = 0.40

# Signal colours — identical to notebook
COLOURS = {
    "MOM_12_1"  : "#2563EB",
    "MOM_3_1"   : "#16A34A",
    "MA_Cross"  : "#EA580C",
    "RSI_14"    : "#7C3AED",
    "Composite" : "#DC2626",
    "ML"        : "#0891B2",
    "Benchmark" : "#94A3B8",
}

SIGNAL_COLS   = ["MOM_12_1", "MOM_3_1", "MA_Cross", "RSI_14"]
SIGNAL_LABELS = {
    "MOM_12_1"  : "Momentum 12-1",
    "MOM_3_1"   : "Momentum 3-1",
    "MA_Cross"  : "MA Crossover",
    "RSI_14"    : "RSI-14",
    "Composite" : "Composite",
}

# Plotly layout — light theme applied to every figure
PLOTLY_BASE = dict(
    paper_bgcolor = "#FAFAF7",
    plot_bgcolor  = "#FAFAF7",
    font          = dict(family="'DM Mono', 'Courier New', monospace",
                         size=11, color="#1C1917"),
    legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                         font=dict(size=10)),
    margin        = dict(l=48, r=16, t=36, b=36),
    hoverlabel    = dict(bgcolor="#1C1917", font_size=11,
                         font_family="'DM Mono', monospace",
                         font_color="#FAFAF7"),
    xaxis         = dict(gridcolor="#E7E5E4", linecolor="#D6D3D1",
                         tickfont=dict(size=10)),
    yaxis         = dict(gridcolor="#E7E5E4", linecolor="#D6D3D1",
                         tickfont=dict(size=10)),
)

def _fig():
    """Return a blank figure with the base layout applied."""
    f = go.Figure()
    f.update_layout(**PLOTLY_BASE)
    return f

def _apply(fig):
    fig.update_layout(**PLOTLY_BASE)
    return fig


# =============================================================================
# 2. DATA LOADERS
# =============================================================================

@lru_cache(maxsize=1)
def get_prices() -> pd.DataFrame:
    """
    Load and cache the wide daily price matrix from DuckDB.
    Applies quality filters matching notebook Section 3.
    Returns wide DataFrame: index=date, columns=ticker.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    raw = con.execute("""
        SELECT date, ticker, close
        FROM   daily_prices
        ORDER  BY ticker, date
    """).df()
    con.close()

    raw["date"] = pd.to_datetime(raw["date"])

    # Check 1: sufficient history
    counts   = raw.groupby("ticker")["date"].count()
    min_days = int(MIN_HISTORY_YEARS * TRADING_DAYS)
    ok       = counts[counts >= min_days].index

    wide = raw.pivot(index="date", columns="ticker", values="close")
    wide = wide[ok]

    # Check 2: no abnormal single-day returns
    rets = wide.pct_change()
    bad  = (rets.abs() > MAX_DAILY_RETURN).any()
    wide = wide[ok.difference(bad[bad].index)]

    # Check 3: no large consecutive gaps (> 7 calendar days)
    ok_tickers = []
    for ticker in wide.columns:
        gaps = wide[ticker].dropna().index.to_series().diff().dt.days
        if gaps.max() <= 7:
            ok_tickers.append(ticker)
    return wide[ok_tickers].sort_index()


@lru_cache(maxsize=1)
def get_benchmark() -> pd.Series:
    """
    Return a daily close price Series for the S&P 500 (^GSPC), indexed by
    date, with name "Benchmark".

    Source priority
    ───────────────
    1. yfinance  — always attempted first; provides accurate, dividend-adjusted
                   S&P 500 total-return index prices directly from Yahoo Finance.
    2. Local CSV cache (benchmark_gspc.csv) — used when yfinance is unavailable
                   (no network, rate-limit).  Cache is refreshed whenever
                   yfinance succeeds.
    3. DuckDB    — last resort: reads ^GSPC rows already stored in daily_prices.
                   Present only if the original download script stored it.
    4. Equal-weighted fallback — if all of the above fail, returns the
                   equal-weighted average log-return of the clean universe
                   (a rough proxy, clearly labelled in the header).

    The returned Series is a price series (not returns).  run_backtest()
    converts it to log-returns at the desired rebalancing frequency.
    """
    # ── 1. Try yfinance ───────────────────────────────────────────────────────
    try:
        raw = yf.download(
            "^GSPC",
            start="2010-01-01",          # generous start — backtest trims later
            auto_adjust=True,            # total-return adjusted prices
            progress=False,
            threads=False,
        )
        if raw is not None and not raw.empty:
            # yfinance may return MultiIndex columns for a single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            s = raw["close"].rename("Benchmark")
            s.index = pd.to_datetime(s.index)
            s = s.dropna().sort_index()
            if len(s) > 100:                 # sanity: got meaningful data
                s.to_csv(BENCHMARK_CACHE)    # refresh cache
                return s
    except Exception:
        pass   # network failure — try cache next

    # ── 2. Local CSV cache ────────────────────────────────────────────────────
    if BENCHMARK_CACHE.exists():
        try:
            s = pd.read_csv(BENCHMARK_CACHE, index_col=0, parse_dates=True)
            # handle both single-column and multi-column cache files
            if isinstance(s, pd.DataFrame):
                col = "close" if "close" in s.columns else s.columns[0]
                s   = s[col]
            s = s.rename("Benchmark").dropna().sort_index()
            if len(s) > 100:
                return s
        except Exception:
            pass

    # ── 3. DuckDB fallback ────────────────────────────────────────────────────
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        df  = con.execute("""
            SELECT date, close FROM daily_prices
            WHERE  ticker = '^GSPC' ORDER BY date
        """).df()
        con.close()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["close"].rename("Benchmark").dropna()
            if len(s) > 100:
                return s
    except Exception:
        pass

    # ── 4. Equal-weighted universe proxy (last resort) ────────────────────────
    prices  = get_prices()
    monthly = prices.resample("ME").last()
    log_ret = np.log(monthly / monthly.shift(1))
    # Return as a "price" series starting at 100 so run_backtest can resample it
    proxy = np.exp(log_ret.mean(axis=1).cumsum()) * 100
    return proxy.rename("Benchmark")


def fetch_ff3() -> pd.DataFrame:
    """Fetch FF3 monthly factors with CSV cache (30-day TTL)."""
    if FF_CACHE.exists():
        age = (date.today() - date.fromtimestamp(FF_CACHE.stat().st_mtime)).days
        if age < 30:
            return pd.read_csv(FF_CACHE, index_col=0, parse_dates=True)

    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_CSV.zip")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            raw = z.read(z.namelist()[0]).decode("utf-8")
    except Exception as exc:
        if FF_CACHE.exists():
            return pd.read_csv(FF_CACHE, index_col=0, parse_dates=True)
        raise RuntimeError(f"Cannot fetch FF3 data: {exc}")

    data_lines = [
        [p.strip() for p in line.split(",")]
        for line in raw.splitlines()
        if len(line.split(",")) == 5
        and line.split(",")[0].strip().isdigit()
        and len(line.split(",")[0].strip()) == 6
    ]

    df = (
        pd.DataFrame(data_lines, columns=["yyyymm", "Mkt_RF", "SMB", "HML", "RF"])
        .assign(
            date   = lambda x: pd.to_datetime(x["yyyymm"].str.strip(), format="%Y%m")
                                + pd.offsets.MonthEnd(0),
            Mkt_RF = lambda x: pd.to_numeric(x["Mkt_RF"], errors="coerce") / 100,
            SMB    = lambda x: pd.to_numeric(x["SMB"],    errors="coerce") / 100,
            HML    = lambda x: pd.to_numeric(x["HML"],    errors="coerce") / 100,
            RF     = lambda x: pd.to_numeric(x["RF"],     errors="coerce") / 100,
        )
        .dropna(subset=["Mkt_RF"])
        .set_index("date")[["Mkt_RF", "SMB", "HML", "RF"]]
    )
    df.to_csv(FF_CACHE)
    return df


# =============================================================================
# 3. SIGNAL ENGINES
# =============================================================================

def z_score_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: zero mean, unit std across tickers each month."""
    return df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)


def compute_signals(prices: pd.DataFrame, lookback: int = 12,
                    freq: str = "ME") -> dict:
    """
    Compute all four signals plus Composite on a wide daily price DataFrame.
    Returns dict: signal_name -> wide DataFrame (period-end × ticker).

    freq controls the rebalancing calendar — "ME" = monthly, "QE" = quarterly.
    MA_Cross and RSI use daily prices for their EMAs and are then resampled to
    freq at the end, so signal quality is preserved regardless of freq.
    """
    periodic = prices.resample(freq).last()

    # MOM_12_1: log return from t-lookback to t-1 (skip most recent period)
    mom12 = np.log(periodic.shift(1) / periodic.shift(1 + lookback))

    # MOM_3_1: log return from t-3 to t-1 (always 3 periods back)
    mom3  = np.log(periodic.shift(1) / periodic.shift(4))

    # MA_Cross: EMA_50 / EMA_200 - 1, computed on daily prices, resampled to freq
    def _ma_cross_ticker(px):
        ema50  = px.ewm(span=50,  adjust=False).mean()
        ema200 = px.ewm(span=200, adjust=False).mean()
        return ((ema50 / ema200) - 1).resample(freq).last()

    ma = (
        prices
        .apply(_ma_cross_ticker)
        .reindex(periodic.index)
    )

    # RSI_14: Wilder EMA smoothing on daily prices, resampled to freq
    def _rsi_ticker(px, period=14):
        delta    = px.diff()
        gain     = delta.clip(lower=0).ewm(span=2*period-1, adjust=False).mean()
        loss     = (-delta.clip(upper=0)).ewm(span=2*period-1, adjust=False).mean()
        rs       = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).resample(freq).last()

    rsi = prices.apply(_rsi_ticker).reindex(periodic.index)

    signals = {
        "MOM_12_1" : mom12,
        "MOM_3_1"  : mom3,
        "MA_Cross" : ma,
        "RSI_14"   : rsi,
    }

    # Composite: equal-weighted average of cross-sectional z-scores
    z_all = {k: z_score_cs(v) for k, v in signals.items()}
    signals["Composite"] = pd.concat(z_all.values()).groupby(level=0).mean()

    return signals


# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================

def run_backtest(
    prices     : pd.DataFrame,
    signal_name: str,
    lookback   : int   = 12,
    freq       : str   = "ME",
    cost_bps   : float = 10.0,
    long_short : bool  = True,
) -> pd.DataFrame:
    """
    Walk-forward backtest.  freq controls rebalancing cadence ("ME" or "QE").
    Returns DataFrame with columns:
        [strategy_return, benchmark_return, cum_strategy, cum_benchmark]

    Fixes applied vs original:
      1. freq is passed into compute_signals so signals and rebal_dates share
         the same calendar grid.
      2. Benchmark timestamps are normalised to period-end before lookup so
         .get() always finds a match (no more all-NaN benchmark_return).
      3. Cumulative returns use exp(cumsum) — correct for log-returns.
    """
    # ── signals on the chosen rebalancing grid ────────────────────────────────
    all_sigs = compute_signals(prices, lookback, freq)   # FIX 1: pass freq
    sig_df   = all_sigs[signal_name]

    periodic = prices.resample(freq).last()
    log_ret  = np.log(periodic / periodic.shift(1))

    # ── benchmark log-returns, index normalised to period-end ─────────────────
    bench_series = get_benchmark()
    if isinstance(bench_series.index[0], pd.Timestamp):
        bench_monthly = (
            bench_series
            .resample(freq).last()
            .pipe(lambda s: np.log(s / s.shift(1)))
            .dropna()
        )
    else:
        bench_monthly = bench_series

    # FIX 2: normalise benchmark index so .get() always resolves
    bench_monthly.index = bench_monthly.index + pd.offsets.MonthEnd(0)

    cost      = cost_bps / 10_000.0
    strat_ret = []
    bench_ret = []
    dates     = []

    rebal_dates = sig_df.index[sig_df.index >= pd.Timestamp(BACKTEST_START)]
    prev_long   = set()
    prev_short  = set()

    for i, rd in enumerate(rebal_dates[:-1]):
        scores = sig_df.loc[rd].dropna()
        if len(scores) < 20:
            continue

        next_rd      = rebal_dates[i + 1]
        next_rd_norm = next_rd + pd.offsets.MonthEnd(0)   # FIX 2: normalise key

        long_thresh  = scores.quantile(1 - TOP_DECILE)
        long_stocks  = set(scores[scores >= long_thresh].index)
        long_ret     = log_ret.loc[next_rd, list(long_stocks)].dropna().mean()

        long_to      = (1.0 if not prev_long else
                        1 - len(long_stocks & prev_long) / max(len(long_stocks), 1))
        net           = long_ret - cost * long_to

        if long_short:
            short_thresh = scores.quantile(BOTTOM_DECILE)
            short_stocks = set(scores[scores <= short_thresh].index)
            short_ret    = log_ret.loc[next_rd, list(short_stocks)].dropna().mean()
            short_to     = (1.0 if not prev_short else
                            1 - len(short_stocks & prev_short) / max(len(short_stocks), 1))
            net = (long_ret - cost * long_to) - (short_ret - cost * short_to)
            prev_short = short_stocks

        prev_long = long_stocks
        strat_ret.append(net)
        bench_ret.append(bench_monthly.get(next_rd_norm, np.nan))   # FIX 2
        dates.append(next_rd)

    if not dates:
        return pd.DataFrame()

    df = pd.DataFrame({
        "strategy_return"  : strat_ret,
        "benchmark_return" : bench_ret,
    }, index=dates).dropna()

    # FIX 3: correct compounding for log-returns → exp(cumsum)
    df["cum_strategy"]  = np.exp(df["strategy_return"].cumsum())
    df["cum_benchmark"] = np.exp(df["benchmark_return"].cumsum())
    return df


# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================

def compute_metrics(bt: pd.DataFrame) -> dict:
    """Standard performance metrics matching notebook Section 8."""
    nan_dict = {k: float("nan") for k in
                ["ann_return", "ann_vol", "sharpe", "max_dd",
                 "calmar", "win_rate"]}
    if bt.empty:
        return nan_dict

    r = bt["strategy_return"].dropna()
    if len(r) == 0:
        return nan_dict

    ann_ret = r.mean() * 12
    ann_vol = r.std()  * np.sqrt(12)
    sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan

    wealth  = np.exp(r.cumsum())   # FIX: correct compounding for log-returns
    peak    = wealth.cummax()
    max_dd  = ((wealth - peak) / peak).min()

    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    win_rate = (r > 0).mean()

    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "calmar": calmar, "win_rate": win_rate}


def compute_bench_metrics(bt: pd.DataFrame) -> dict:
    """Same metrics computed on the benchmark column."""
    nan_dict = {k: float("nan") for k in
                ["ann_return", "ann_vol", "sharpe", "max_dd",
                 "calmar", "win_rate"]}
    if bt.empty or "benchmark_return" not in bt.columns:
        return nan_dict

    r = bt["benchmark_return"].dropna()
    if len(r) == 0:
        return nan_dict

    ann_ret = r.mean() * 12
    ann_vol = r.std()  * np.sqrt(12)
    sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan

    wealth  = np.exp(r.cumsum())   # FIX: correct compounding for log-returns
    peak    = wealth.cummax()
    max_dd  = ((wealth - peak) / peak).min()

    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    win_rate = (r > 0).mean()

    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "calmar": calmar, "win_rate": win_rate}


# =============================================================================
# 6. FAMA-FRENCH REGRESSION
# =============================================================================

def ols_regression(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS regression — numpy only, no statsmodels dependency."""
    n, k  = X.shape
    beta  = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    s2    = (resid @ resid) / (n - k)
    se    = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t_stat  = beta / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
    r2      = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return {"beta": beta, "se": se, "t_stat": t_stat,
            "p_value": p_value, "r_squared": r2}


def run_ff3(bt: pd.DataFrame, ff: pd.DataFrame) -> dict:
    """Run FF3 OLS regression on strategy excess returns."""
    if bt.empty or ff.empty:
        return {}

    monthly = bt["strategy_return"].dropna()
    ff_m    = ff.copy()
    ff_m.index = pd.to_datetime(ff_m.index) + pd.offsets.MonthEnd(0)
    monthly.index = pd.to_datetime(monthly.index) + pd.offsets.MonthEnd(0)

    aligned = monthly.to_frame("ret").join(ff_m, how="inner").dropna()
    if len(aligned) < 12:
        return {}

    y = (aligned["ret"] - aligned["RF"]).values
    X = np.column_stack([
        np.ones(len(aligned)),
        aligned["Mkt_RF"].values,
        aligned["SMB"].values,
        aligned["HML"].values,
    ])

    res = ols_regression(y, X)

    return {
        "alpha_annual"  : res["beta"][0] * 12,
        "alpha_t"       : res["t_stat"][0],
        "market_beta"   : res["beta"][1],
        "smb_beta"      : res["beta"][2],
        "hml_beta"      : res["beta"][3],
        "market_se"     : res["se"][1],
        "smb_se"        : res["se"][2],
        "hml_se"        : res["se"][3],
        "r2"            : res["r_squared"],
        "n_months"      : len(aligned),
        "aligned"       : aligned,
    }


def rolling_alpha(bt: pd.DataFrame, ff: pd.DataFrame, window: int = 12) -> pd.Series:
    """Rolling window FF3 alpha (annualised)."""
    if bt.empty or ff.empty:
        return pd.Series(dtype=float)

    monthly = bt["strategy_return"].dropna()
    ff_m    = ff.copy()
    ff_m.index = pd.to_datetime(ff_m.index) + pd.offsets.MonthEnd(0)
    monthly.index = pd.to_datetime(monthly.index) + pd.offsets.MonthEnd(0)

    aligned = monthly.to_frame("ret").join(ff_m, how="inner").dropna()
    aligned["excess"] = aligned["ret"] - aligned["RF"]

    alphas, idx = [], []
    for end in range(window, len(aligned) + 1):
        chunk = aligned.iloc[end - window : end]
        X_    = np.column_stack([np.ones(len(chunk)),
                                  chunk[["Mkt_RF", "SMB", "HML"]].values])
        try:
            coef  = np.linalg.lstsq(X_, chunk["excess"].values, rcond=None)[0]
            alphas.append(coef[0] * 12)
        except Exception:
            alphas.append(np.nan)
        idx.append(aligned.index[end - 1])

    return pd.Series(alphas, index=idx)


# =============================================================================
# 7. SCREENER
# =============================================================================

def build_screener(prices: pd.DataFrame, signal_name: str) -> pd.DataFrame:
    """Build current momentum screener as of most recent month."""
    all_sigs = compute_signals(prices, lookback=12)
    sig      = all_sigs[signal_name]

    latest   = sig.iloc[-1].dropna().sort_values(ascending=False)
    monthly  = prices.resample("ME").last()

    ret_1m   = monthly.pct_change(1).iloc[-1]
    ret_3m   = monthly.pct_change(3).iloc[-1]
    ret_12m  = (monthly.shift(1) / monthly.shift(13) - 1).iloc[-1]

    rsi_now  = z_score_cs(all_sigs["RSI_14"]).iloc[-1]
    ma_now   = z_score_cs(all_sigs["MA_Cross"]).iloc[-1]
    mom12_z  = z_score_cs(all_sigs["MOM_12_1"]).iloc[-1]
    mom3_z   = z_score_cs(all_sigs["MOM_3_1"]).iloc[-1]

    tickers  = latest.index
    df = pd.DataFrame({
        "Ticker"       : tickers,
        "Signal Score" : latest.values.round(3),
        "MOM 12-1"     : mom12_z.reindex(tickers).values.round(3),
        "MOM 3-1"      : mom3_z.reindex(tickers).values.round(3),
        "MA Cross"     : ma_now.reindex(tickers).values.round(3),
        "RSI-14"       : rsi_now.reindex(tickers).values.round(3),
        "1M Ret %"     : (ret_1m.reindex(tickers).values * 100).round(2),
        "3M Ret %"     : (ret_3m.reindex(tickers).values * 100).round(2),
        "12M Ret %"    : (ret_12m.reindex(tickers).values * 100).round(2),
    }).reset_index(drop=True)

    df.insert(0, "Rank", range(1, len(df) + 1))
    n = len(df)
    df["Decile"] = pd.cut(df["Rank"], bins=10, labels=range(1, 11)).astype(int)
    return df


# =============================================================================
# 8. ML ENGINE
# =============================================================================

def _stack_wide_to_long(wide_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Safely stack a wide (period × ticker) DataFrame to long format.
    Works with pandas ≥ 2.0 where reset_index() no longer yields
    'level_0'/'level_1' but the actual index names instead.
    Returns DataFrame with columns: ['month', 'ticker', value_name].
    """
    s = wide_df.stack(future_stack=True) if hasattr(pd.DataFrame, "stack") else wide_df.stack()
    df = s.reset_index()
    # Rename whatever the first two columns are → 'month', 'ticker'
    cols = list(df.columns)
    df.columns = ["month", "ticker"] + cols[2:]
    if len(df.columns) == 3:
        df.columns = ["month", "ticker", value_name]
    return df


def run_ml(prices: pd.DataFrame, model_name: str = "LightGBM",
           split: float = 0.70) -> dict:
    """
    Train and evaluate a binary classifier on momentum signals.

    Dataset
    -------
    Features : cross-sectional z-scores of all four signals at month t
    Label    : 1 if the stock lands in the top decile of next-month returns

    Random-walk sanity check
    -------------------------
    The same model architecture is trained on a synthetic dataset where
    prices follow a pure random walk (no momentum).  AUC near 0.50 on
    this dataset confirms the model is not just memorising noise.

    Returns dict with keys:
        auc_real, f1_real, auc_rw, feat_imp,
        ml_cum (wealth index), comp_bt (Composite backtest DataFrame),
        prob (test-set probabilities), labels (test-set true labels),
        model_name, train_period, test_period
    """
    all_sigs = compute_signals(prices, lookback=12)
    monthly  = prices.resample("ME").last()
    log_ret  = np.log(monthly / monthly.shift(1))

    # ── Build real-data ML dataset ────────────────────────────────────────────
    z_sigs = {name: z_score_cs(df) for name, df in all_sigs.items()
              if name in SIGNAL_COLS}

    # Stack each signal wide→long and concat side-by-side
    sig_parts = []
    for name, df in z_sigs.items():
        part = _stack_wide_to_long(df, name)[["month", "ticker", name]]
        sig_parts.append(part.set_index(["month", "ticker"]))
    signal_long = (
        pd.concat(sig_parts, axis=1)
        .reset_index()                       # columns: month, ticker, sig1, sig2 …
    )

    # Forward returns: return at t+1, attached to the signal row at t
    fwd_long = _stack_wide_to_long(log_ret, "fwd_ret")
    fwd_long["month"] = fwd_long.groupby("ticker")["month"].shift(-1)
    fwd_long = fwd_long.dropna(subset=["month"])

    ml_df = (
        signal_long
        .merge(fwd_long, on=["month", "ticker"], how="inner")
        .dropna()
        .assign(
            label=lambda x: (
                x.groupby("month")["fwd_ret"]
                 .transform(lambda r: (r >= r.quantile(1 - TOP_DECILE)).astype(int))
            )
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    # ── Build random-walk dataset (sanity check) ───────────────────────────────
    # Prices follow iid normal draws — momentum signals should carry zero
    # predictive power here.  AUC ≈ 0.50 validates that the real-data result
    # is not an artefact of the model architecture.
    np.random.seed(42)
    rw_log_ret = pd.DataFrame(
        np.random.normal(log_ret.mean().mean(), log_ret.std().mean(), log_ret.shape),
        index=log_ret.index, columns=log_ret.columns,
    )
    # Correct compounding for log-returns: exp(cumsum) not (1+r).cumprod()
    rw_price = np.exp(rw_log_ret.cumsum())

    rw_cols = ["MOM_12_1", "MOM_3_1"]
    rw_sig_wide = {
        "MOM_12_1": np.log(rw_price.shift(1) / rw_price.shift(13)),
        "MOM_3_1" : np.log(rw_price.shift(1) / rw_price.shift(4)),
    }
    rw_z = {k: z_score_cs(v) for k, v in rw_sig_wide.items()}

    rw_sig_parts = []
    for name, df in rw_z.items():
        part = _stack_wide_to_long(df, name)[["month", "ticker", name]]
        rw_sig_parts.append(part.set_index(["month", "ticker"]))
    rw_signal_long = pd.concat(rw_sig_parts, axis=1).reset_index()

    rw_fwd = _stack_wide_to_long(rw_log_ret, "fwd_ret")
    rw_fwd["month"] = rw_fwd.groupby("ticker")["month"].shift(-1)
    rw_fwd = rw_fwd.dropna(subset=["month"])

    rw_df = (
        rw_signal_long
        .merge(rw_fwd, on=["month", "ticker"], how="inner")
        .dropna()
        .assign(
            label=lambda x: (
                x.groupby("month")["fwd_ret"]
                 .transform(lambda r: (r >= r.quantile(1 - TOP_DECILE)).astype(int))
            )
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    # ── Train / test split (strict temporal) ─────────────────────────────────
    split_idx  = int(len(ml_df) * split)
    train_df   = ml_df.iloc[:split_idx]
    test_df    = ml_df.iloc[split_idx:]

    feat_cols  = SIGNAL_COLS
    X_tr = train_df[feat_cols].values;  y_tr = train_df["label"].values
    X_te = test_df[feat_cols].values;   y_te = test_df["label"].values

    # Scale — required for Logistic Regression; harmless for trees
    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    train_period = (train_df["month"].min(), train_df["month"].max())
    test_period  = (test_df["month"].min(),  test_df["month"].max())

    # Random walk split (same temporal fraction)
    rw_split = int(len(rw_df) * split)
    rw_train = rw_df.iloc[:rw_split];  rw_test = rw_df.iloc[rw_split:]
    rX_tr = rw_train[rw_cols].values;  ry_tr = rw_train["label"].values
    rX_te = rw_test[rw_cols].values;   ry_te = rw_test["label"].values
    rsc     = StandardScaler()
    rX_tr_s = rsc.fit_transform(rX_tr)
    rX_te_s = rsc.transform(rX_te)

    # ── Model zoo ─────────────────────────────────────────────────────────────
    # Each entry: (real_model, rw_model, X_train, X_test, rX_train, rX_test)
    models_map = {
        "Logistic Regression": (
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
            X_tr_s, X_te_s, rX_tr_s, rX_te_s,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            X_tr, X_te, rX_tr, rX_te,
        ),
        "LightGBM": (
            lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1),
            X_tr, X_te, rX_tr, rX_te,
        ),
    }

    mdl, rw_mdl, Xtr, Xte, rXtr, rXte = models_map[model_name]

    # Real data
    mdl.fit(Xtr, y_tr)
    prob     = mdl.predict_proba(Xte)[:, 1]
    pred     = mdl.predict(Xte)
    auc_real = roc_auc_score(y_te, prob)
    f1_real  = f1_score(y_te, pred)

    # Random walk
    rw_mdl.fit(rXtr, ry_tr)
    rw_prob = rw_mdl.predict_proba(rXte)[:, 1]
    auc_rw  = roc_auc_score(ry_te, rw_prob)

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(mdl, "feature_importances_"):
        fi = pd.Series(mdl.feature_importances_, index=feat_cols)
    elif hasattr(mdl, "coef_"):
        fi = pd.Series(np.abs(mdl.coef_[0]), index=feat_cols)
    else:
        fi = pd.Series(np.zeros(len(feat_cols)), index=feat_cols)
    fi = fi.sort_values()

    # ── ML backtest (long-only, with transaction cost) ────────────────────────
    ml_cost      = TRANSACTION_COST_BPS / 10_000.0
    ml_sig_long  = test_df[["month", "ticker"]].assign(ml_prob=prob)
    ml_sig_wide  = ml_sig_long.pivot(index="month", columns="ticker", values="ml_prob")

    common       = log_ret.index.intersection(ml_sig_wide.index)
    bt_ml        = []
    bt_dates     = []
    prev_long_ml = set()

    for i, rd in enumerate(common[:-1]):
        scores = ml_sig_wide.loc[rd].dropna()
        if len(scores) < 20:
            continue
        lt        = scores.quantile(1 - TOP_DECILE)
        holdings  = set(scores[scores >= lt].index)
        nret      = log_ret.loc[common[i + 1], list(holdings)].dropna().mean()
        turnover  = (1.0 if not prev_long_ml else
                     1 - len(holdings & prev_long_ml) / max(len(holdings), 1))
        bt_ml.append(nret - ml_cost * turnover)
        bt_dates.append(common[i + 1])
        prev_long_ml = holdings

    ml_cum = (np.exp(pd.Series(bt_ml, index=bt_dates).cumsum())
              if bt_dates else pd.Series(dtype=float))

    # Composite (long-only) for side-by-side comparison
    comp_bt = run_backtest(prices, "Composite", 12, "ME", 10, False)

    return {
        "auc_real"    : auc_real,
        "f1_real"     : f1_real,
        "auc_rw"      : auc_rw,
        "feat_imp"    : fi,
        "ml_cum"      : ml_cum,
        "comp_bt"     : comp_bt,
        "prob"        : prob,
        "labels"      : y_te,
        "model_name"  : model_name,
        "train_period": train_period,
        "test_period" : test_period,
    }


# =============================================================================
# 9. STYLE TOKENS
# =============================================================================

# Colour palette
C_BG      = "#FAFAF7"     # page background
C_SURFACE = "#FFFFFF"     # card surface
C_BORDER  = "#E7E5E4"     # card borders
C_TEXT    = "#1C1917"     # primary text
C_MUTED   = "#78716C"     # secondary / label text
C_ACCENT  = "#2563EB"     # interactive accent (blue)
C_GREEN   = "#16A34A"
C_RED     = "#DC2626"
C_AMBER   = "#D97706"

FONT_MONO  = "'DM Mono', 'Courier New', monospace"
FONT_SERIF = "'DM Serif Display', Georgia, serif"

# Shared component styles
SIDEBAR_STYLE = {
    "width"          : "240px",
    "minWidth"       : "240px",
    "backgroundColor": C_SURFACE,
    "borderRight"    : f"1px solid {C_BORDER}",
    "padding"        : "20px 16px",
    "overflowY"      : "auto",
    "height"         : "calc(100vh - 52px)",
}

MAIN_STYLE = {
    "flex"          : "1",
    "backgroundColor": C_BG,
    "padding"       : "16px 20px",
    "overflowY"     : "auto",
    "height"        : "calc(100vh - 52px)",
}

CARD_STYLE = {
    "backgroundColor": C_SURFACE,
    "border"         : f"1px solid {C_BORDER}",
    "borderRadius"   : "6px",
    "padding"        : "12px 14px",
}

KPI_STYLE = {
    **CARD_STYLE,
    "textAlign" : "center",
    "padding"   : "10px 8px",
}

LABEL_S = {
    "fontSize"     : "9px",
    "letterSpacing": "0.12em",
    "color"        : C_MUTED,
    "fontFamily"   : FONT_MONO,
    "textTransform": "uppercase",
    "marginBottom" : "2px",
}

VALUE_S = {
    "fontSize"  : "20px",
    "fontWeight": "600",
    "fontFamily": FONT_MONO,
    "color"     : C_TEXT,
    "lineHeight": "1.2",
}

CTRL_LABEL_S = {
    "fontSize"     : "9px",
    "letterSpacing": "0.1em",
    "color"        : C_MUTED,
    "fontFamily"   : FONT_MONO,
    "textTransform": "uppercase",
    "marginTop"    : "14px",
    "marginBottom" : "4px",
    "display"      : "flex",
    "alignItems"   : "center",
    "gap"          : "4px",
}

BTN_STYLE = {
    "width"          : "100%",
    "backgroundColor": C_ACCENT,
    "color"          : "#fff",
    "border"         : "none",
    "borderRadius"   : "4px",
    "padding"        : "9px",
    "fontSize"       : "11px",
    "fontFamily"     : FONT_MONO,
    "cursor"         : "pointer",
    "marginTop"      : "18px",
    "letterSpacing"  : "0.08em",
}

DROPDOWN_STYLE = {
    "backgroundColor": C_SURFACE,
    "border"         : f"1px solid {C_BORDER}",
    "borderRadius"   : "4px",
    "fontSize"       : "11px",
    "fontFamily"     : FONT_MONO,
}

RADIO_STYLE = {
    "fontSize"  : "11px",
    "fontFamily": FONT_MONO,
    "color"     : C_TEXT,
}

SECTION_LABEL_S = {
    "fontSize"     : "8px",
    "letterSpacing": "0.18em",
    "color"        : C_MUTED,
    "fontFamily"   : FONT_MONO,
    "textTransform": "uppercase",
    "borderBottom" : f"1px solid {C_BORDER}",
    "paddingBottom": "6px",
    "marginBottom" : "10px",
}

# =============================================================================
# 10. UI HELPER COMPONENTS
# =============================================================================

_tip_count = [0]

def info_tip(text: str) -> html.Span:
    """ⓘ icon with hover tooltip."""
    _tip_count[0] += 1
    tid = f"tip-{_tip_count[0]}"
    return html.Span([
        html.Span("ⓘ", id=tid, style={
            "fontSize"  : "11px",
            "color"     : C_MUTED,
            "cursor"    : "help",
            "marginLeft": "4px",
        }),
        dbc.Tooltip(text, target=tid, placement="top",
                    style={"maxWidth": "280px", "fontSize": "11px",
                           "lineHeight": "1.5", "fontFamily": FONT_MONO}),
    ], style={"display": "inline"})


def ctrl_label(text: str, tip: str = "") -> html.Div:
    children = [html.Span(text)]
    if tip:
        children.append(info_tip(tip))
    return html.Div(children, style=CTRL_LABEL_S)


def run_btn(btn_id: str, label: str = "▶  RUN") -> html.Button:
    return html.Button(label, id=btn_id, n_clicks=0, style=BTN_STYLE)


def signal_dd(dd_id: str, default: str = "MOM_12_1") -> dcc.Dropdown:
    return dcc.Dropdown(
        id=dd_id,
        options=[{"label": v, "value": k} for k, v in SIGNAL_LABELS.items()],
        value=default, clearable=False, style=DROPDOWN_STYLE,
    )


def lookback_sl(sl_id: str) -> dcc.Slider:
    return dcc.Slider(id=sl_id, min=3, max=12, step=None,
                      marks={3:"3M", 6:"6M", 9:"9M", 12:"12M"}, value=12,
                      tooltip={"always_visible": False})


def cost_sl(sl_id: str) -> dcc.Slider:
    return dcc.Slider(id=sl_id, min=0, max=50, step=5,
                      marks={i: str(i) for i in range(0, 51, 10)}, value=10,
                      tooltip={"placement": "bottom", "always_visible": True})


def portfolio_radio(r_id: str) -> dcc.RadioItems:
    return dcc.RadioItems(
        id=r_id,
        options=[{"label": " Long Only",    "value": "lo"},
                 {"label": " Long / Short", "value": "ls"}],
        value="lo",
        labelStyle={**RADIO_STYLE, "marginRight": "12px"},
    )


def chart_title(label: str, tip: str = "") -> html.Div:
    """Small chart section title with optional ⓘ."""
    children = [html.Span(label, style={
        "fontSize"     : "10px",
        "letterSpacing": "0.1em",
        "color"        : C_MUTED,
        "fontFamily"   : FONT_MONO,
        "textTransform": "uppercase",
    })]
    if tip:
        children.append(info_tip(tip))
    return html.Div(children, style={"marginBottom": "4px"})


def kpi_card(label: str, val_id: str, tip: str = "") -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div([html.Span(label, style=LABEL_S), info_tip(tip) if tip else ""],
                 style={"display": "flex", "alignItems": "center",
                        "justifyContent": "center", "gap": "2px"}),
        html.Div("—", id=val_id, style=VALUE_S),
    ], style=KPI_STYLE), xs=12, sm=6, md=True)


def graph(g_id: str, height: str = "220px") -> dcc.Graph:
    return dcc.Graph(
        id=g_id,
        config={"displayModeBar": False},
        style={"height": height},
        figure=_fig(),
    )


def sidebar_header(text: str) -> html.Div:
    return html.Div(text, style=SECTION_LABEL_S)


# =============================================================================
# 11. APP LAYOUT
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Serif+Display&display=swap",
    ],
    title="Pythia — Momentum Dashboard",
    suppress_callback_exceptions=True,
)

# Expose the Flask server so gunicorn can find it:
#   gunicorn pythia_dashboard:server
server = app.server

# ── Global CSS ────────────────────────────────────────────────────────────────
app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  body { background:#FAFAF7; margin:0; padding:0; }
  * { box-sizing:border-box; }
  .dash-dropdown .Select-control { background:#fff !important; border-color:#E7E5E4 !important; }
  .dash-dropdown .Select-menu-outer { background:#fff !important; border-color:#E7E5E4 !important; }
  .dash-dropdown .Select-value-label { color:#1C1917 !important; font-family:'DM Mono',monospace !important; font-size:11px !important; }
  .dash-dropdown .Select-option { font-family:'DM Mono',monospace !important; font-size:11px !important; color:#1C1917 !important; }
  .dash-dropdown .Select-option.is-focused { background:#F5F5F4 !important; }
  .rc-slider-rail { background:#E7E5E4 !important; }
  .rc-slider-track { background:#2563EB !important; }
  .rc-slider-handle { border-color:#2563EB !important; }
  .nav-tabs .nav-link { font-family:'DM Mono',monospace !important; font-size:11px !important; letter-spacing:0.08em; color:#78716C !important; border:none !important; border-bottom:2px solid transparent !important; padding:10px 16px !important; text-transform:uppercase; }
  .nav-tabs .nav-link.active { color:#2563EB !important; border-bottom:2px solid #2563EB !important; background:transparent !important; }
  .nav-tabs { border-bottom:1px solid #E7E5E4 !important; }
  ::-webkit-scrollbar { width:4px; height:4px; }
  ::-webkit-scrollbar-track { background:#F5F5F4; }
  ::-webkit-scrollbar-thumb { background:#D6D3D1; border-radius:2px; }
  input[type=radio] { accent-color:#2563EB; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

# ── Header ────────────────────────────────────────────────────────────────────
header = html.Div(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("PYTHIA", style={
                    "fontSize": "15px", "fontWeight": "600",
                    "color": C_ACCENT, "fontFamily": FONT_MONO,
                    "letterSpacing": "0.2em",
                }),
                html.Span("  ·  Momentum Strategy Dashboard", style={
                    "fontSize": "11px", "color": C_MUTED,
                    "fontFamily": FONT_MONO,
                }),
            ]), width="auto"),
            dbc.Col(html.Span("S&P 500  ·  10Y Daily OHLCV  ·  DuckDB", style={
                "fontSize": "10px", "color": C_BORDER,
                "fontFamily": FONT_MONO, "float": "right",
                "lineHeight": "28px", "letterSpacing": "0.08em",
            })),
        ], align="center"),
    ], fluid=True),
    style={
        "backgroundColor": C_TEXT,
        "borderBottom"   : f"1px solid #292524",
        "padding"        : "11px 0",
    }
)

# =============================================================================
# TAB 1 — STRATEGY BACKTESTER
# =============================================================================

tab1_sidebar = html.Div([
    sidebar_header("Backtest Parameters"),

    ctrl_label("Signal Strategy",
               "Choose which momentum signal ranks stocks each month. "
               "Composite blends all four signals into a single z-scored rank."),
    signal_dd("t1-signal"),

    ctrl_label("Portfolio Type",
               "Long/Short buys top decile and shorts bottom decile. "
               "Long Only buys top decile only."),
    portfolio_radio("t1-ls"),

    ctrl_label("Rebalancing Frequency",
               "How often the portfolio is reconstituted. "
               "Monthly captures more signal updates but incurs higher costs."),
    dcc.RadioItems(
        id="t1-freq",
        options=[{"label": " Monthly",   "value": "ME"},
                 {"label": " Quarterly", "value": "QE"}],
        value="ME",
        labelStyle={**RADIO_STYLE, "marginRight": "12px"},
    ),

    ctrl_label("Transaction Cost (bps)",
               "One-way cost per trade. "
               "10 bps is a typical institutional estimate for large-cap S&P 500 stocks."),
    cost_sl("t1-cost"),

    run_btn("t1-run", "▶  RUN BACKTEST"),
], style=SIDEBAR_STYLE)


tab1_main = html.Div([
    # ── Strategy KPI strip ────────────────────────────────────────────────────
    html.Div("STRATEGY", style={**SECTION_LABEL_S, "marginBottom": "6px"}),
    dbc.Row([
        kpi_card("Ann. Return",  "t1-kpi-ret",
                 "Geometric mean annual return over the full backtest period."),
        kpi_card("Sharpe Ratio", "t1-kpi-sharpe",
                 "Annualised excess return divided by annualised volatility. "
                 "Values above 1.0 are generally considered strong."),
        kpi_card("Max Drawdown", "t1-kpi-dd",
                 "Largest peak-to-trough decline over the backtest. "
                 "Measures worst-case loss the strategy experienced."),
        kpi_card("Calmar Ratio", "t1-kpi-calmar",
                 "Annualised return divided by absolute max drawdown. "
                 "Rewards strategies that earn well relative to their worst loss."),
        kpi_card("Win Rate",     "t1-kpi-win",
                 "Fraction of months with a positive strategy return. "
                 "Values above 55% indicate consistent performance."),
    ], className="g-2 mb-2"),

    # ── Benchmark KPI strip ───────────────────────────────────────────────────
    html.Div("BENCHMARK (S&P 500)", style={**SECTION_LABEL_S, "marginBottom": "6px"}),
    dbc.Row([
        kpi_card("Ann. Return",  "t1-bkpi-ret",
                 "Benchmark annualised return over the same backtest period."),
        kpi_card("Sharpe Ratio", "t1-bkpi-sharpe",
                 "Benchmark Sharpe ratio — compare with strategy above."),
        kpi_card("Max Drawdown", "t1-bkpi-dd",
                 "Benchmark worst peak-to-trough decline."),
        kpi_card("Calmar Ratio", "t1-bkpi-calmar",
                 "Benchmark Calmar ratio."),
        kpi_card("Win Rate",     "t1-bkpi-win",
                 "Fraction of periods with a positive benchmark return."),
    ], className="g-2 mb-2"),

    # 2x2 chart grid
    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Cumulative Return",
                        "Portfolio value of $1 invested at inception. "
                        "Shaded area highlights periods of outperformance vs benchmark."),
            graph("t1-cum", "230px"),
        ], style=CARD_STYLE), md=6),

        dbc.Col(html.Div([
            chart_title("Rolling 12-Month Sharpe",
                        "Sharpe ratio computed over a trailing 12-month window. "
                        "Sustained positive values indicate consistent risk-adjusted performance."),
            graph("t1-sharpe", "230px"),
        ], style=CARD_STYLE), md=6),
    ], className="g-2 mb-2"),

    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Drawdown Profile",
                        "How far below its previous peak the strategy sits at each point in time. "
                        "Deep prolonged drawdowns are worse for investors than short sharp drops."),
            graph("t1-dd", "200px"),
        ], style=CARD_STYLE), md=6),

        dbc.Col(html.Div([
            chart_title("Rolling Volatility",
                        "Annualised standard deviation of monthly returns over a trailing 12-month window. "
                        "Shows how the strategy's risk level has changed over time."),
            graph("t1-vol", "200px"),
        ], style=CARD_STYLE), md=6),
    ], className="g-2"),
], style=MAIN_STYLE)


# =============================================================================
# TAB 2 — STOCK SCREENER
# =============================================================================

tab2_sidebar = html.Div([
    sidebar_header("Screener Parameters"),

    ctrl_label("Ranking Signal",
               "Signal used to rank stocks. Composite blends all four."),
    signal_dd("t2-signal", "Composite"),

    ctrl_label("Show Top N Stocks",
               "Number of top and bottom stocks displayed in the bar chart."),
    dcc.Slider(id="t2-topn", min=10, max=50, step=None,
               marks={10:"10", 20:"20", 30:"30", 50:"50"}, value=20,
               tooltip={"always_visible": False}),

    run_btn("t2-run", "▶  REFRESH"),

    html.Div(id="t2-date", style={
        "marginTop": "20px", "fontSize": "10px",
        "color": C_MUTED, "fontFamily": FONT_MONO,
    }),
], style=SIDEBAR_STYLE)


tab2_main = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Composite Score — Top & Bottom",
                        "Horizontal bars show composite momentum z-score. "
                        "Green = long candidates (high momentum), Red = short candidates (low momentum)."),
            graph("t2-bars", "460px"),
        ], style=CARD_STYLE), md=5),

        dbc.Col(html.Div([
            chart_title("Full Universe Ranking",
                        "All clean-universe stocks ranked by signal score. "
                        "Top decile highlighted green, bottom decile red. Sortable."),
            html.Div(
                id="t2-table",
                style={
                    "height"   : "460px",
                    "overflowY": "auto",
                    "fontSize" : "10px",
                    "fontFamily": FONT_MONO,
                }
            ),
        ], style=CARD_STYLE), md=7),
    ], className="g-2"),
], style=MAIN_STYLE)


# =============================================================================
# TAB 3 — SIGNAL COMPARISON
# =============================================================================

tab3_sidebar = html.Div([
    sidebar_header("Comparison Parameters"),

    ctrl_label("Portfolio Type",
               "Applies to all signals simultaneously."),
    portfolio_radio("t3-ls"),

    ctrl_label("Transaction Cost (bps)",
               "Same cost applied to all signals for a fair comparison."),
    cost_sl("t3-cost"),

    run_btn("t3-run", "▶  RUN ALL SIGNALS"),
], style=SIDEBAR_STYLE)


tab3_main = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Cumulative Return — All Signals",
                        "One line per signal plus benchmark. "
                        "Solid = Long Only, Dashed = Long-Short. "
                        "Colour matches signal identity throughout the dashboard."),
            graph("t3-cum", "260px"),
        ], style=CARD_STYLE), md=12),
    ], className="g-2 mb-2"),

    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Performance Metrics",
                        "One row per signal plus benchmark. "
                        "Best value per column highlighted green, worst red."),
            html.Div(id="t3-table", style={"fontFamily": FONT_MONO, "fontSize": "10px"}),
        ], style=CARD_STYLE), md=6),

        dbc.Col(html.Div([
            chart_title("Sharpe vs Calmar",
                        "Sharpe measures return per unit of volatility. "
                        "Calmar measures return per unit of worst drawdown. "
                        "A strategy that scores well on both is robust."),
            graph("t3-bars", "260px"),
        ], style=CARD_STYLE), md=6),
    ], className="g-2"),
], style=MAIN_STYLE)


# =============================================================================
# TAB 4 — FAMA-FRENCH
# =============================================================================

tab4_sidebar = html.Div([
    sidebar_header("Regression Parameters"),

    ctrl_label("Signal Strategy",
               "Strategy whose returns are decomposed into factor exposures."),
    signal_dd("t4-signal"),

    ctrl_label("Portfolio Type",
               "Long/Short strips out most market beta, making alpha cleaner. "
               "Long Only retains high market exposure."),
    portfolio_radio("t4-ls"),

    run_btn("t4-run", "▶  RUN REGRESSION"),
], style=SIDEBAR_STYLE)


tab4_main = html.Div([
    # KPI strip
    dbc.Row([
        kpi_card("Ann. Alpha",  "t4-alpha",
                 "Annual return unexplained by market, size, and value factors. "
                 "Positive alpha means the strategy earns above what factor exposure predicts."),
        kpi_card("Alpha t-stat", "t4-tstat",
                 "Statistical significance of alpha. "
                 "Values above 2.0 indicate significance at the 95% confidence level."),
        kpi_card("R²",           "t4-r2",
                 "Fraction of strategy return variance explained by the three factors. "
                 "Higher R² means more of the return is explained by systematic risk."),
    ], className="g-2 mb-2"),

    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Factor Loadings",
                        "How much each factor contributes to the strategy return. "
                        "Error bars show 95% confidence intervals. "
                        "Market β near zero in long-short strategies is expected."),
            graph("t4-factors", "280px"),
        ], style=CARD_STYLE), md=6),

        dbc.Col(html.Div([
            chart_title("Rolling 12-Month Alpha",
                        "How alpha has evolved over time. "
                        "Consistent positive alpha is more credible than a single lucky period. "
                        "Green shading = positive alpha, red = negative."),
            graph("t4-rolling", "280px"),
        ], style=CARD_STYLE), md=6),
    ], className="g-2"),
], style=MAIN_STYLE)


# =============================================================================
# TAB 5 — MACHINE LEARNING
# =============================================================================

tab5_sidebar = html.Div([
    sidebar_header("ML Parameters"),

    ctrl_label("Model",
               "Logistic Regression is the interpretable baseline. "
               "Random Forest and LightGBM capture non-linear signal interactions."),
    dcc.Dropdown(
        id="t5-model",
        options=[{"label": m, "value": m} for m in
                 ["Logistic Regression", "Random Forest", "LightGBM"]],
        value="LightGBM", clearable=False, style=DROPDOWN_STYLE,
    ),

    ctrl_label("Train / Test Split",
               "Temporal split: training on earlier months, testing on later months. "
               "No future data is used in training."),
    dcc.RadioItems(
        id="t5-split",
        options=[{"label": " 70 / 30", "value": "0.70"},
                 {"label": " 80 / 20", "value": "0.80"}],
        value="0.70",
        labelStyle={**RADIO_STYLE, "marginRight": "12px"},
    ),

    run_btn("t5-run", "▶  RUN MODEL"),
], style=SIDEBAR_STYLE)


tab5_main = html.Div([
    # KPI strip
    dbc.Row([
        kpi_card("AUC-ROC (Real)",    "t5-auc",
                 "Area under the ROC curve on real data test set. "
                 "Values above 0.55 indicate the model finds genuine predictive structure."),
        kpi_card("F1 Score (Real)",   "t5-f1",
                 "Harmonic mean of precision and recall on the test set. "
                 "Measures how well the model identifies top-decile stocks."),
        kpi_card("AUC (Random Walk)", "t5-auc-rw",
                 "AUC on a synthetic random walk dataset with no momentum. "
                 "Should be near 0.50. Values above 0.52 suggest overfitting."),
    ], className="g-2 mb-2"),

    # 2x2 chart grid
    dbc.Row([
        dbc.Col(html.Div([
            chart_title("ML vs Composite Cumulative Return",
                        "ML backtest uses the model's predicted top-decile probability as the ranking signal, "
                        "compared to the simple Composite signal. Both long-only for a clean comparison."),
            graph("t5-cum", "230px"),
        ], style=CARD_STYLE), md=6),

        dbc.Col(html.Div([
            chart_title("Feature Importance",
                        "Which signals the model relies on most. "
                        "For tree models: split count. For Logistic Regression: absolute coefficient."),
            graph("t5-imp", "230px"),
        ], style=CARD_STYLE), md=6),
    ], className="g-2 mb-2"),

    dbc.Row([
        dbc.Col(html.Div([
            chart_title("Real Data vs Random Walk AUC",
                        "The gap between real data AUC and random walk AUC confirms the signals "
                        "carry genuine predictive structure. A well-specified model should produce "
                        "AUC near 0.50 on random walk data."),
            graph("t5-auc-bar", "200px"),
        ], style=CARD_STYLE), md=6),

        dbc.Col(html.Div([
            chart_title("Predicted Probability Distribution",
                        "Distribution of model confidence scores split by actual outcome. "
                        "Green = stocks that actually landed in the top decile. "
                        "A well-calibrated model shows green distribution shifted right."),
            graph("t5-prob", "200px"),
        ], style=CARD_STYLE), md=6),
    ], className="g-2"),
], style=MAIN_STYLE)


# =============================================================================
# 12. FULL LAYOUT ASSEMBLY
# =============================================================================

tabs = dbc.Tabs([
    dbc.Tab(label="Strategy Backtester", tab_id="t1",
            children=html.Div([tab1_sidebar, tab1_main],
                               style={"display": "flex", "flexDirection": "row"})),
    dbc.Tab(label="Stock Screener",      tab_id="t2",
            children=html.Div([tab2_sidebar, tab2_main],
                               style={"display": "flex", "flexDirection": "row"})),
    dbc.Tab(label="Signal Comparison",   tab_id="t3",
            children=html.Div([tab3_sidebar, tab3_main],
                               style={"display": "flex", "flexDirection": "row"})),
    dbc.Tab(label="Fama-French",         tab_id="t4",
            children=html.Div([tab4_sidebar, tab4_main],
                               style={"display": "flex", "flexDirection": "row"})),
    dbc.Tab(label="Machine Learning",    tab_id="t5",
            children=html.Div([tab5_sidebar, tab5_main],
                               style={"display": "flex", "flexDirection": "row"})),
], active_tab="t1", style={"backgroundColor": C_BG})

app.layout = html.Div([
    header,
    tabs,
], style={"backgroundColor": C_BG, "minHeight": "100vh"})


# =============================================================================
# 13. CALLBACKS
# =============================================================================

# Metric formatter helpers
def fmt_pct(v):
    if v is None or np.isnan(v): return "—"
    return f"{v*100:+.1f}%"

def fmt_2f(v):
    if v is None or np.isnan(v): return "—"
    return f"{v:.2f}"

def kpi_color(v, good_above=0, neutral=False):
    if v is None or np.isnan(v): return {**VALUE_S, "color": C_MUTED}
    if neutral:                   return {**VALUE_S, "color": C_TEXT}
    color = C_GREEN if v > good_above else C_RED
    return {**VALUE_S, "color": color}


# ── Tab 1 callback ────────────────────────────────────────────────────────────

@app.callback(
    Output("t1-cum",         "figure"),
    Output("t1-sharpe",      "figure"),
    Output("t1-dd",          "figure"),
    Output("t1-vol",         "figure"),
    # Strategy KPIs
    Output("t1-kpi-ret",    "children"), Output("t1-kpi-ret",    "style"),
    Output("t1-kpi-sharpe", "children"), Output("t1-kpi-sharpe", "style"),
    Output("t1-kpi-dd",     "children"), Output("t1-kpi-dd",     "style"),
    Output("t1-kpi-calmar", "children"), Output("t1-kpi-calmar", "style"),
    Output("t1-kpi-win",    "children"), Output("t1-kpi-win",    "style"),
    # Benchmark KPIs
    Output("t1-bkpi-ret",    "children"), Output("t1-bkpi-ret",    "style"),
    Output("t1-bkpi-sharpe", "children"), Output("t1-bkpi-sharpe", "style"),
    Output("t1-bkpi-dd",     "children"), Output("t1-bkpi-dd",     "style"),
    Output("t1-bkpi-calmar", "children"), Output("t1-bkpi-calmar", "style"),
    Output("t1-bkpi-win",    "children"), Output("t1-bkpi-win",    "style"),
    Input("t1-run", "n_clicks"),
    State("t1-signal",   "value"),
    State("t1-ls",       "value"),
    State("t1-freq",     "value"),
    State("t1-cost",     "value"),
    prevent_initial_call=False,
)
def update_tab1(n, signal, ls, freq, cost):
    prices     = get_prices()
    long_short = (ls == "ls")
    signal     = signal or "MOM_12_1"
    lookback   = 12
    freq       = freq or "ME"
    cost       = float(cost or 10)

    bt   = run_backtest(prices, signal, lookback, freq, cost, long_short)
    mets = compute_metrics(bt)
    col  = COLOURS.get(signal, C_ACCENT)

    empty = [_fig()] * 4 + ["—", VALUE_S] * 10   # 5 strategy + 5 benchmark KPIs
    if bt.empty:
        return tuple(empty)

    # ── Cumulative return chart ───────────────────────────────────────────────
    fig_cum = _fig()
    fig_cum.add_trace(go.Scatter(
        x=bt.index, y=bt["cum_strategy"], name=SIGNAL_LABELS.get(signal, signal),
        line=dict(color=col, width=2),
        hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
    ))
    fig_cum.add_trace(go.Scatter(
        x=bt.index, y=bt["cum_benchmark"], name="Benchmark",
        line=dict(color=COLOURS["Benchmark"], width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
    ))
    # Shade outperformance region
    fig_cum.add_trace(go.Scatter(
        x=pd.concat([bt.index.to_series(), bt.index.to_series()[::-1]]).values,
        y=pd.concat([bt["cum_strategy"], bt["cum_benchmark"][::-1]]).values,
        fill="toself", fillcolor="rgba(37,99,235,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_cum.add_hline(y=1, line=dict(color=C_BORDER, width=1))
    fig_cum.update_layout(**PLOTLY_BASE, title=None)

    # ── Rolling 12-month Sharpe ───────────────────────────────────────────────
    roll_s = (bt["strategy_return"]
              .rolling(12)
              .apply(lambda r: (r.mean()*12 - RF_ANNUAL) / (r.std()*np.sqrt(12))))
    fig_sr = _fig()
    fig_sr.add_trace(go.Scatter(
        x=roll_s.index, y=roll_s,
        line=dict(color=col, width=1.8),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig_sr.add_hline(y=0, line=dict(color=C_MUTED, width=1, dash="dash"))
    fig_sr.update_layout(**PLOTLY_BASE)

    # ── Drawdown profile ──────────────────────────────────────────────────────
    # cum_strategy and cum_benchmark are already exp(cumsum) so drawdown is exact
    wealth  = bt["cum_strategy"]
    dd      = (wealth - wealth.cummax()) / wealth.cummax()
    bench_w = bt["cum_benchmark"]
    bench_d = (bench_w - bench_w.cummax()) / bench_w.cummax()
    fig_dd  = _fig()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd, fill="tozeroy",
        fillcolor="rgba(220,38,38,0.10)",
        line=dict(color=C_RED, width=1.2),
        name=SIGNAL_LABELS.get(signal, signal),
        hovertemplate="%{x|%b %Y}<br>%{y:.1%}<extra></extra>",
    ))
    fig_dd.add_trace(go.Scatter(
        x=bench_d.index, y=bench_d,
        line=dict(color=COLOURS["Benchmark"], width=1, dash="dot"),
        name="Benchmark",
        hovertemplate="%{x|%b %Y}<br>%{y:.1%}<extra></extra>",
    ))
    fig_dd.update_yaxes(tickformat=".0%")
    fig_dd.update_layout(**PLOTLY_BASE)

    # ── Rolling volatility ────────────────────────────────────────────────────
    roll_vol = (bt["strategy_return"].rolling(12).std() * np.sqrt(12) * 100)
    fig_vol  = _fig()
    fig_vol.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol,
        line=dict(color=col, width=1.8),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="%{x|%b %Y}<br>Vol: %{y:.1f}%<extra></extra>",
    ))
    fig_vol.update_yaxes(ticksuffix="%")
    fig_vol.update_layout(**PLOTLY_BASE)

    # ── Strategy KPIs ─────────────────────────────────────────────────────────
    ann_ret = mets["ann_return"]
    sharpe  = mets["sharpe"]
    max_dd  = mets["max_dd"]
    calmar  = mets["calmar"]
    win     = mets["win_rate"]

    # ── Benchmark KPIs ────────────────────────────────────────────────────────
    bench_m      = compute_bench_metrics(bt)
    bench_ret    = bench_m.get("ann_return", np.nan)
    bench_sharpe = bench_m.get("sharpe",     np.nan)
    bench_dd     = bench_m.get("max_dd",     np.nan)
    bench_calmar = bench_m.get("calmar",     np.nan)
    bench_win    = bench_m.get("win_rate",   np.nan)

    # Colour thresholds: strategy beats benchmark → green, lags → red
    _b_ret    = bench_ret    if not np.isnan(bench_ret)    else 0
    _b_sharpe = bench_sharpe if not np.isnan(bench_sharpe) else 0
    _b_dd     = bench_dd     if not np.isnan(bench_dd)     else -0.20
    _b_calmar = bench_calmar if not np.isnan(bench_calmar) else 0
    _b_win    = bench_win    if not np.isnan(bench_win)    else 0.50

    return (
        fig_cum, fig_sr, fig_dd, fig_vol,
        # Strategy KPIs — coloured relative to benchmark
        fmt_pct(ann_ret), kpi_color(ann_ret, _b_ret),
        fmt_2f(sharpe),   kpi_color(sharpe,  _b_sharpe),
        fmt_pct(max_dd),  kpi_color(max_dd,  _b_dd),
        fmt_2f(calmar),   kpi_color(calmar,  _b_calmar),
        f"{win:.0%}",     kpi_color(win,     _b_win),
        # Benchmark KPIs — displayed in muted colour (no comparison needed)
        fmt_pct(bench_ret),    {**VALUE_S, "color": C_MUTED},
        fmt_2f(bench_sharpe),  {**VALUE_S, "color": C_MUTED},
        fmt_pct(bench_dd),     {**VALUE_S, "color": C_MUTED},
        fmt_2f(bench_calmar),  {**VALUE_S, "color": C_MUTED},
        f"{bench_win:.0%}" if not np.isnan(bench_win) else "—",
                               {**VALUE_S, "color": C_MUTED},
    )


# ── Tab 2 callback ────────────────────────────────────────────────────────────

@app.callback(
    Output("t2-bars",  "figure"),
    Output("t2-table", "children"),
    Output("t2-date",  "children"),
    Input("t2-run",    "n_clicks"),
    State("t2-signal", "value"),
    State("t2-topn",   "value"),
    prevent_initial_call=False,
)
def update_tab2(n, signal, topn):
    prices  = get_prices()
    signal  = signal or "Composite"
    topn    = int(topn or 20)

    screener    = build_screener(prices, signal)
    top_n       = screener.head(topn)
    bottom_n    = screener.tail(topn).sort_values("Signal Score")
    latest_date = prices.resample("ME").last().index[-1].strftime("%B %Y")

    # Bar chart
    plot_df  = pd.concat([top_n, bottom_n]).drop_duplicates("Ticker")
    colours  = [C_GREEN if v >= 0 else C_RED for v in plot_df["Signal Score"]]
    fig_bars = _fig()
    fig_bars.add_trace(go.Bar(
        x=plot_df["Signal Score"],
        y=plot_df["Ticker"],
        orientation="h",
        marker_color=colours,
        marker_opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
    ))
    fig_bars.add_vline(x=0, line=dict(color=C_BORDER, width=1))
    fig_bars.update_yaxes(autorange="reversed")
    fig_bars.update_layout(**PLOTLY_BASE, bargap=0.2)

    # Ranked table
    display_cols = ["Rank", "Ticker", "Signal Score",
                    "MOM 12-1", "MOM 3-1", "MA Cross", "RSI-14",
                    "1M Ret %", "3M Ret %", "12M Ret %"]
    display_df = screener[display_cols].copy()

    header = html.Tr([
        html.Th(c, style={
            "fontSize": "9px", "color": C_MUTED, "fontFamily": FONT_MONO,
            "padding": "5px 8px", "borderBottom": f"1px solid {C_BORDER}",
            "letterSpacing": "0.08em", "textTransform": "uppercase",
            "position": "sticky", "top": "0",
            "backgroundColor": C_SURFACE, "whiteSpace": "nowrap",
        }) for c in display_cols
    ])

    def row_bg(decile):
        if decile <= 1:  return "rgba(22,163,74,0.07)"
        if decile >= 10: return "rgba(220,38,38,0.07)"
        return "transparent"

    rows = [
        html.Tr([
            html.Td(str(val), style={
                "fontSize": "10px", "fontFamily": FONT_MONO,
                "padding": "4px 8px", "color": C_TEXT,
                "borderBottom": f"1px solid {C_BORDER}",
                "whiteSpace": "nowrap",
            }) for val in row
        ], style={"backgroundColor": row_bg(int(screener.iloc[i]["Decile"]))})
        for i, row in enumerate(display_df.values.tolist())
    ]

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    date_label = f"Screener as of  {latest_date}"
    return fig_bars, table, date_label


# ── Tab 3 callback ────────────────────────────────────────────────────────────

@app.callback(
    Output("t3-cum",   "figure"),
    Output("t3-bars",  "figure"),
    Output("t3-table", "children"),
    Input("t3-run",      "n_clicks"),
    State("t3-ls",       "value"),
    State("t3-cost",     "value"),
    prevent_initial_call=False,
)
def update_tab3(n, ls, cost):
    prices     = get_prices()
    long_short = (ls == "ls")
    lookback   = 12
    cost       = float(cost or 10)

    results = {}
    metrics_map = {}
    for sig in SIGNAL_COLS + ["Composite"]:
        bt = run_backtest(prices, sig, lookback, "ME", cost, long_short)
        results[sig]      = bt
        metrics_map[sig]  = compute_metrics(bt)

    bench_bt   = next(iter(results.values()))
    bench_mets = compute_bench_metrics(bench_bt)

    # Cumulative return
    fig_cum = _fig()
    for sig, bt in results.items():
        if bt.empty: continue
        ls_str = "LS" if long_short else "LO"
        fig_cum.add_trace(go.Scatter(
            x=bt.index, y=bt["cum_strategy"],
            name=f"{SIGNAL_LABELS[sig]} ({ls_str})",
            line=dict(color=COLOURS[sig], width=1.8,
                      dash="dash" if long_short else "solid"),
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        ))

    if not bench_bt.empty:
        fig_cum.add_trace(go.Scatter(
            x=bench_bt.index, y=bench_bt["cum_benchmark"],
            name="Benchmark",
            line=dict(color=COLOURS["Benchmark"], width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        ))
    fig_cum.update_layout(**PLOTLY_BASE)

    sharpes = [metrics_map[s]["sharpe"] for s in results]
    calmars = [metrics_map[s]["calmar"] for s in results]

    fig_bars = _fig()
    fig_bars.add_trace(go.Bar(
        name="Sharpe", x=[SIGNAL_LABELS[s] for s in results],
        y=sharpes, marker_color=C_ACCENT, opacity=0.85,
        hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig_bars.add_trace(go.Bar(
        name="Calmar", x=[SIGNAL_LABELS[s] for s in results],
        y=calmars, marker_color=C_GREEN, opacity=0.85,
        hovertemplate="%{x}<br>Calmar: %{y:.2f}<extra></extra>",
    ))
    fig_bars.add_hline(y=0, line=dict(color=C_BORDER, width=1))
    fig_bars.update_layout(**PLOTLY_BASE, barmode="group", bargap=0.25)

    # Metrics table
    col_keys   = ["ann_return", "sharpe", "max_dd", "calmar", "win_rate"]
    col_labels = ["Ann. Return", "Sharpe", "Max DD", "Calmar", "Win Rate"]
    col_fmt    = [fmt_pct, fmt_2f, fmt_pct, fmt_2f,
                  lambda v: f"{v:.0%}" if v is not None and not np.isnan(v) else "—"]

    # Collect all values for best/worst highlighting
    all_vals = {}
    for ck in col_keys:
        row_vals = []
        for sig in results:
            row_vals.append(metrics_map[sig].get(ck, np.nan))
        row_vals.append(bench_mets.get(ck, np.nan))
        all_vals[ck] = row_vals

    header_tr = html.Tr(
        [html.Th("Signal", style={
            "fontSize": "9px", "color": C_MUTED, "fontFamily": FONT_MONO,
            "padding": "5px 8px", "borderBottom": f"1px solid {C_BORDER}",
            "letterSpacing": "0.08em", "textTransform": "uppercase",
        })] + [
            html.Th(lbl, style={
                "fontSize": "9px", "color": C_MUTED, "fontFamily": FONT_MONO,
                "padding": "5px 8px", "borderBottom": f"1px solid {C_BORDER}",
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "textAlign": "right",
            }) for lbl in col_labels
        ]
    )

    body_rows = []
    for i, sig in enumerate(list(results.keys()) + ["Benchmark"]):
        is_bench = (sig == "Benchmark")
        mets     = bench_mets if is_bench else metrics_map[sig]
        sig_col  = COLOURS.get("Benchmark" if is_bench else sig, C_TEXT)

        cells = [html.Td(
            "Benchmark" if is_bench else SIGNAL_LABELS[sig],
            style={"fontSize": "10px", "fontFamily": FONT_MONO,
                   "padding": "5px 8px", "color": sig_col,
                   "borderBottom": f"1px solid {C_BORDER}",
                   "borderTop": f"1px solid {C_BORDER}" if is_bench else "none"},
        )]

        for j, (ck, fmt) in enumerate(zip(col_keys, col_fmt)):
            v     = mets.get(ck, np.nan)
            vals  = [x for x in all_vals[ck] if not np.isnan(x)]
            best  = max(vals) if vals else np.nan
            worst = min(vals) if vals else np.nan
            if not np.isnan(v) and not np.isnan(best):
                vcol = (C_GREEN if abs(v - best) < 1e-9
                        else (C_RED if abs(v - worst) < 1e-9 else C_TEXT))
            else:
                vcol = C_MUTED
            cells.append(html.Td(
                fmt(v),
                style={"fontSize": "10px", "fontFamily": FONT_MONO,
                       "padding": "5px 8px", "color": vcol,
                       "borderBottom": f"1px solid {C_BORDER}",
                       "borderTop": f"1px solid {C_BORDER}" if is_bench else "none",
                       "textAlign": "right"},
            ))
        body_rows.append(html.Tr(cells))

    table = html.Table(
        [html.Thead(header_tr), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    return fig_cum, fig_bars, table


# ── Tab 4 callback ────────────────────────────────────────────────────────────

@app.callback(
    Output("t4-factors", "figure"),
    Output("t4-rolling", "figure"),
    Output("t4-alpha",   "children"), Output("t4-alpha",   "style"),
    Output("t4-tstat",   "children"), Output("t4-tstat",   "style"),
    Output("t4-r2",      "children"), Output("t4-r2",      "style"),
    Input("t4-run",      "n_clicks"),
    State("t4-signal",   "value"),
    State("t4-ls",       "value"),
    prevent_initial_call=False,
)
def update_tab4(n, signal, ls):
    prices     = get_prices()
    signal     = signal or "MOM_12_1"
    long_short = (ls == "ls")
    lookback   = 12

    bt  = run_backtest(prices, signal, lookback, "ME", TRANSACTION_COST_BPS, long_short)
    ff  = fetch_ff3()
    reg = run_ff3(bt, ff)

    na_style = {**VALUE_S, "color": C_MUTED}
    if not reg:
        return _fig(), _fig(), "—", na_style, "—", na_style, "—", na_style

    alpha  = reg["alpha_annual"]
    t_stat = reg["alpha_t"]
    r2     = reg["r2"]

    # Factor loadings bar chart
    factors = ["Alpha (ann.)", "Market β", "Size β (SMB)", "Value β (HML)"]
    values  = [alpha * 100,
               reg["market_beta"], reg["smb_beta"], reg["hml_beta"]]
    errors  = [0, reg["market_se"] * 1.96,
               reg["smb_se"] * 1.96, reg["hml_se"] * 1.96]
    bar_cols = [C_GREEN if v >= 0 else C_RED for v in values]

    fig_fac = _fig()
    fig_fac.add_trace(go.Bar(
        x=values, y=factors, orientation="h",
        marker_color=bar_cols, marker_opacity=0.85,
        error_x=dict(type="data", array=errors,
                     color=C_MUTED, thickness=1.5, width=4),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig_fac.add_vline(x=0, line=dict(color=C_BORDER, width=1))
    fig_fac.update_layout(**PLOTLY_BASE)

    # Rolling alpha
    roll_a = rolling_alpha(bt, ff, window=12)
    fig_roll = _fig()
    if not roll_a.empty:
        fig_roll.add_trace(go.Scatter(
            x=roll_a.index, y=roll_a * 100,
            line=dict(color=C_ACCENT, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(22,163,74,0.08)",
            hovertemplate="%{x|%b %Y}<br>Alpha: %{y:.2f}%<extra></extra>",
            name="Rolling Alpha",
        ))
        fig_roll.add_hline(y=0, line=dict(color=C_RED, width=1, dash="dash"))
        fig_roll.update_yaxes(ticksuffix="%")
    fig_roll.update_layout(**PLOTLY_BASE)

    return (
        fig_fac, fig_roll,
        f"{alpha*100:+.2f}%",   kpi_color(alpha, 0),
        f"{t_stat:.2f}",        kpi_color(t_stat, 2.0),
        f"{r2:.3f}",            {**VALUE_S, "color": C_TEXT},
    )


# ── Tab 5 callback ────────────────────────────────────────────────────────────

@app.callback(
    Output("t5-cum",     "figure"),
    Output("t5-imp",     "figure"),
    Output("t5-auc-bar", "figure"),
    Output("t5-prob",    "figure"),
    Output("t5-auc",     "children"), Output("t5-auc",     "style"),
    Output("t5-f1",      "children"), Output("t5-f1",      "style"),
    Output("t5-auc-rw",  "children"), Output("t5-auc-rw",  "style"),
    Input("t5-run",    "n_clicks"),
    State("t5-model",  "value"),
    State("t5-split",  "value"),
    prevent_initial_call=False,
)
def update_tab5(n, model_name, split):
    prices     = get_prices()
    model_name = model_name or "LightGBM"
    split      = float(split or 0.70)

    res = run_ml(prices, model_name, split)

    # ── Cumulative return: ML vs Composite ────────────────────────────────────
    train_end  = res["train_period"][1]
    test_start = res["test_period"][0]
    test_end   = res["test_period"][1]
    period_note = (f"Test window: {test_start.strftime('%b %Y')} – "
                   f"{test_end.strftime('%b %Y')}")

    fig_cum = _fig()
    if not res["ml_cum"].empty:
        fig_cum.add_trace(go.Scatter(
            x=res["ml_cum"].index, y=res["ml_cum"],
            name=f"ML ({model_name})",
            line=dict(color=COLOURS["ML"], width=2),
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
        ))
    comp_bt = res["comp_bt"]
    if not comp_bt.empty:
        # Trim Composite to test window so curves start at the same point
        comp_test = comp_bt[comp_bt.index >= test_start]
        if not comp_test.empty:
            # Re-base to 1.0 at start of test period
            comp_rebased = comp_test["cum_strategy"] / comp_test["cum_strategy"].iloc[0]
            fig_cum.add_trace(go.Scatter(
                x=comp_rebased.index, y=comp_rebased,
                name="Composite (LO)",
                line=dict(color=COLOURS["Composite"], width=1.8, dash="dot"),
                hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra></extra>",
            ))
    fig_cum.add_hline(y=1, line=dict(color=C_BORDER, width=1))
    fig_cum.update_layout(**PLOTLY_BASE,
                          title=dict(text=period_note,
                                     font=dict(size=9, color=C_MUTED), x=0))

    # ── Feature importance ────────────────────────────────────────────────────
    fi      = res["feat_imp"]
    fig_imp = _fig()
    bar_cols = [COLOURS.get(s, C_ACCENT) for s in fi.index]
    fig_imp.add_trace(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker_color=bar_cols, marker_opacity=0.85,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_imp.update_layout(**PLOTLY_BASE)

    # ── Real vs Random Walk AUC ───────────────────────────────────────────────
    auc_real = res["auc_real"]
    auc_rw   = res["auc_rw"]
    y_max    = min(1.0, max(auc_real, auc_rw, 0.55) + 0.06)

    fig_auc = _fig()
    fig_auc.add_trace(go.Bar(
        name="Real Data",
        x=[model_name], y=[auc_real],
        marker_color=C_ACCENT, opacity=0.85,
        hovertemplate="%{x}<br>AUC: %{y:.3f}<extra></extra>",
        text=[f"{auc_real:.3f}"], textposition="outside",
    ))
    fig_auc.add_trace(go.Bar(
        name="Random Walk",
        x=[model_name], y=[auc_rw],
        marker_color=COLOURS["Benchmark"], opacity=0.85,
        hovertemplate="%{x}<br>AUC: %{y:.3f}<extra></extra>",
        text=[f"{auc_rw:.3f}"], textposition="outside",
    ))
    fig_auc.add_hline(y=0.5, line=dict(color=C_RED, width=1, dash="dash"),
                      annotation_text="Random (0.50)", annotation_position="right")
    fig_auc.update_layout(**PLOTLY_BASE, barmode="group", bargap=0.35)
    fig_auc.update_yaxes(range=[0.44, y_max])

    # ── Predicted probability distribution ────────────────────────────────────
    fig_prob = _fig()
    prob   = res["prob"]
    labels = res["labels"]
    if len(prob) > 0:
        p_pos = prob[labels == 1]
        p_neg = prob[labels == 0]
        fig_prob.add_trace(go.Histogram(
            x=p_neg, xbins=dict(start=0, end=1, size=1/29),
            name="Not top decile (label=0)",
            marker_color=COLOURS["Benchmark"], opacity=0.6,
            histnorm="probability density",
            hovertemplate="Prob: %{x:.2f}<br>Density: %{y:.2f}<extra></extra>",
        ))
        fig_prob.add_trace(go.Histogram(
            x=p_pos, xbins=dict(start=0, end=1, size=1/29),
            name="Top decile (label=1)",
            marker_color=C_GREEN, opacity=0.7,
            histnorm="probability density",
            hovertemplate="Prob: %{x:.2f}<br>Density: %{y:.2f}<extra></extra>",
        ))
        fig_prob.update_layout(**PLOTLY_BASE, barmode="overlay")
        fig_prob.update_xaxes(title_text="Predicted Probability", range=[0, 1])

    # ── KPIs ──────────────────────────────────────────────────────────────────
    f1_real = res["f1_real"]

    return (
        fig_cum, fig_imp, fig_auc, fig_prob,
        f"{auc_real:.3f}", kpi_color(auc_real, 0.55),
        f"{f1_real:.3f}",  kpi_color(f1_real,  0.15),
        f"{auc_rw:.3f}",   kpi_color(auc_rw, 0.52, neutral=True),
    )


# =============================================================================
# 14. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os
    port  = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"

    print("PYTHIA  —  Momentum Strategy Dashboard")
    print(f"DB path   : {DB_PATH.resolve()}")
    print(f"FF cache  : {FF_CACHE.resolve()}")
    print(f"Port      : {port}")

    if not DB_PATH.exists():
        print(f"WARNING: {DB_PATH} not found.")
        print("Run download_sp500_to_duckdb.py first.")

    app.run(debug=debug, port=port, host="0.0.0.0")
