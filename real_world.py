# Real-data pipeline:
# 1) Download prices (yfinance) for two tickers
# 2) Build spread X = S1 - B*S2 either by EMRT (paper) or OLS (cointegration slope)
# 3) Check stationarity: ADF p-value and OU half-life
# 4) Train Q-learning policy on OU simulations
# 5) Backtest on the real spread
#
# Usage:
#   pip install yfinance statsmodels pandas numpy matplotlib
#   python run_real_pair.py
#

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from main import (
    RLConfig,
    build_spread_with_emrt,
    train_q_on_ou,
    backtest_policy_on_spread,
)

# ----------------------------
# Config — pick a realistic pair
# ----------------------------
TICKER_1 = "KO"    # Coca-Cola
TICKER_2 = "PEP"   # PepsiCo
START = "2012-01-01"
FORMATION_END = "2023-12-30"   # formation window end (inclusive)
TEST_START = "2024-01-03"      # trading/backtest start
TEST_END = None                # None means "to latest"

# Transaction cost per in/out on the spread
TRANS_COST = 0.0025  # e.g., 25 bps per entry/exit; tune to your venue

# ----------------------------
# Helpers
# ----------------------------
def download_prices(t1, t2, start):
    df = yf.download([t1, t2], start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        # Edge case if only one column returned
        df = df.to_frame()
    df = df.dropna().copy()
    # Ensure consistent column order
    df = df[[t1, t2]]
    return df.rename(columns={t1: "S1", t2: "S2"})

def adf_report(x: pd.Series, name: str):
    x = x.dropna()
    res = adfuller(x, autolag="AIC")
    out = {
        "name": name,
        "ADF_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags": int(res[2]),
        "n_obs": int(res[3]),
    }
    return out

def ols_hedge_ratio(s1: pd.Series, s2: pd.Series, add_const=True) -> float:
    """
    Cointegration-style slope: s1 ~ const + beta * s2
    Return beta so spread X = s1 - beta*s2. If add_const, OLS has intercept.
    """
    y = s1.values
    X = s2.values
    X = sm.add_constant(X) if add_const else X.reshape(-1, 1)
    model = sm.OLS(y, X, hasconst=add_const).fit()
    beta = model.params[-1]
    return float(beta)

def ou_half_life(x: pd.Series) -> float:
    """
    Standard half-life estimator:
    ΔX_t = α + β * X_{t-1} + ε
    Half-life = -ln(2) / ln(1 + β)
    """
    x = x.dropna()
    dx = x.diff().dropna()
    x_lag = x.shift(1).dropna().reindex(dx.index)

    X = sm.add_constant(x_lag.values)
    model = sm.OLS(dx.values, X).fit()
    beta = model.params[1]
    hl = -np.log(2) / np.log(1.0 + beta) if (1.0 + beta) > 0 else np.inf
    return float(hl)

def plot_equity(series: pd.Series, title="Equity curve"):
    ax = series.plot(figsize=(10, 4))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 1) Download & prep
# ----------------------------
prices = download_prices(TICKER_1, TICKER_2, START)
s1, s2 = prices["S1"], prices["S2"]

formation_slice = slice(prices.index.min(), pd.to_datetime(FORMATION_END))
test_slice = slice(pd.to_datetime(TEST_START), pd.to_datetime(TEST_END) if TEST_END else prices.index.max())

print(f"Data range: {prices.index.min().date()} → {prices.index.max().date()}")
print(f"Formation window: {formation_slice.start.date()} → {FORMATION_END}")
print(f"Test window: {TEST_START} → {test_slice.stop.date() if isinstance(test_slice.stop, pd.Timestamp) else 'latest'}")

# ----------------------------
# 2) Build spreads (EMRT and OLS)
# ----------------------------
# (A) EMRT-selected hedge ratio (from the paper)
X_emrt, B_emrt, r_form = build_spread_with_emrt(s1, s2, formation_slice, C=2.0, grid_step=0.02)
print(f"[EMRT] B={B_emrt:.4f}, EMRT(formation)={r_form:.4f}")

# (B) OLS hedge ratio as cointegration slope (with intercept)
B_ols = ols_hedge_ratio(s1.loc[formation_slice], s2.loc[formation_slice], add_const=True)
X_ols = s1 - B_ols * s2
print(f"[OLS]  B={B_ols:.4f}")

# Stationarity checks on formation window
adf_emrt = adf_report(X_emrt.loc[formation_slice], "X_emrt (formation)")
adf_ols  = adf_report(X_ols.loc[formation_slice],  "X_ols  (formation)")
hl_emrt  = ou_half_life(X_emrt.loc[formation_slice])
hl_ols   = ou_half_life(X_ols.loc[formation_slice])

print("\nADF (formation window)")
print(f"  {adf_emrt['name']}: ADF={adf_emrt['ADF_stat']:.3f}, p={adf_emrt['p_value']:.4f}")
print(f"  {adf_ols['name']}:  ADF={adf_ols['ADF_stat']:.3f}, p={adf_ols['p_value']:.4f}")
print(f"Half-life (formation) — EMRT: {hl_emrt:.2f} days, OLS: {hl_ols:.2f} days\n")

# Choose one spread for trading. EMRT is aligned with the paper; OLS is the classic CI residual.
USE = "OLS"  # switch to "OLS" to use OLS slope
X = X_emrt if USE.upper() == "EMRT" else X_ols

# ----------------------------
# 3) Train the RL policy on OU sims
# ----------------------------
cfg = RLConfig(
    lookback_l=3,       # slightly smaller state helps coverage
    use_zscore_state=True,
    k_std=0.25,         # try 0.15–0.5
    alpha=0.1, gamma=0.995, epsilon=0.1,
    trans_cost=0.0005,  # start small so behavior shows up
    seed=123
)
agent = train_q_on_ou(num_paths=5000, n_steps=252, cfg=cfg)
# ----------------------------
# 4) Backtest on real spread
# ----------------------------
theta_hat = X.loc[formation_slice].mean()  # reward uses theta; paper uses known theta in sim; here we estimate
bt = backtest_policy_on_spread(X.loc[test_slice], agent, cfg, theta_estimate=float(theta_hat))

print("Backtest metrics on real spread:")
for k, v in bt.metrics.items():
    print(f"  {k:<12} {v:.6f}")

# Optional: quick visuals
plot_equity(bt.equity_curve, title=f"Equity curve ({TICKER_1}-{TICKER_2} spread via {USE})")

# Inspect first few trades
print("\nFirst trades:")
print(bt.trades.head(10).to_string(index=False))
