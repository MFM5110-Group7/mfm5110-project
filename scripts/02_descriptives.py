"""Descriptive statistics, stationarity and white-noise / ARCH tests.

Outputs
-------
outputs/tables/table01_descriptives.{csv,md}
outputs/tables/table02_adf.{csv,md}
outputs/tables/table03_serial_dependence.{csv,md}
outputs/figures/fig01_prices.png
outputs/figures/fig02_returns.png
outputs/figures/fig03_acf_pacf_<code>.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED, OUT_FIGURES, OUT_TABLES  # noqa: E402
from src.plots import COLORS, label, panel_series, save, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def descriptives(ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in ret.columns:
        x = ret[code].dropna()
        jb, jb_p = stats.jarque_bera(x)
        rows.append({
            "index": code,
            "name": label(code),
            "N": len(x),
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "skew": x.skew(),
            "excess_kurt": x.kurtosis(),
            "min": x.min(),
            "max": x.max(),
            "JB": jb,
            "JB_p": jb_p,
            "JB_sig": stars(jb_p),
        })
    return pd.DataFrame(rows).set_index("index")


def adf_table(prices: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in prices.columns:
        for series_name, series in (("price", prices[code]), ("return", ret[code])):
            regs = ("c", "ct") if series_name == "price" else ("c", "ct")
            for reg in regs:
                x = series.dropna().astype(float).values
                nobs = len(x)
                maxlag = int(np.floor(12 * (nobs / 100.0) ** 0.25))
                res = adfuller(x, maxlag=maxlag, regression=reg, autolag="AIC")
                stat, p, usedlag, n_used, crit, icbest = res
                rows.append({
                    "index": code,
                    "series": series_name,
                    "regression": "constant" if reg == "c" else "constant+trend",
                    "lag_AIC": int(usedlag),
                    "stat": stat,
                    "p_value": p,
                    "crit_1pct": crit["1%"],
                    "crit_5pct": crit["5%"],
                    "n_obs": int(n_used),
                    "stationary_5pct": bool(p < 0.05),
                    "sig": stars(p),
                })
    return pd.DataFrame(rows)


def serial_dependence(ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in ret.columns:
        x = ret[code].dropna().astype(float)
        e = x - x.mean()
        lb = acorr_ljungbox(e, lags=[10, 20], return_df=True)
        lb2 = acorr_ljungbox(e ** 2, lags=[10, 20], return_df=True)
        lm_stat, lm_p, f_stat, f_p = het_arch(e.values, nlags=10)
        row = {"index": code, "name": label(code)}
        for lag in (10, 20):
            row[f"LB_r_Q{lag}"] = lb.loc[lag, "lb_stat"]
            row[f"LB_r_p{lag}"] = lb.loc[lag, "lb_pvalue"]
            row[f"LB_r2_Q{lag}"] = lb2.loc[lag, "lb_stat"]
            row[f"LB_r2_p{lag}"] = lb2.loc[lag, "lb_pvalue"]
        row.update({
            "ARCHLM10_stat": lm_stat,
            "ARCHLM10_p": lm_p,
            "ARCHLM10_F": f_stat,
            "ARCHLM10_Fp": f_p,
        })
        rows.append(row)
    return pd.DataFrame(rows).set_index("index")


def acf_figures(ret: pd.DataFrame) -> None:
    style()
    for code in ret.columns:
        x = ret[code].dropna().astype(float)
        e = x - x.mean()
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
        plot_acf(e, lags=30, ax=axes[0, 0], color=COLORS.get(code, "black"), zero=False)
        axes[0, 0].set_title(f"ACF: returns - {label(code)}")
        plot_pacf(e, lags=30, ax=axes[0, 1], method="ywm", color=COLORS.get(code, "black"), zero=False)
        axes[0, 1].set_title("PACF: returns")
        plot_acf(e ** 2, lags=30, ax=axes[1, 0], color=COLORS.get(code, "black"), zero=False)
        axes[1, 0].set_title("ACF: squared returns")
        plot_pacf(e ** 2, lags=30, ax=axes[1, 1], method="ywm", color=COLORS.get(code, "black"), zero=False)
        axes[1, 1].set_title("PACF: squared returns")
        save(fig, OUT_FIGURES / f"fig03_acf_pacf_{code}.png")


def main() -> None:
    prices = pd.read_csv(DATA_PROCESSED / "prices_wide.csv", parse_dates=["date"]).set_index("date")
    ret = pd.read_csv(DATA_PROCESSED / "returns_wide.csv", parse_dates=["date"], index_col="date")

    desc = descriptives(ret)
    desc.to_csv(OUT_TABLES / "table01_descriptives.csv", encoding="utf-8-sig")
    (OUT_TABLES / "table01_descriptives.md").write_text(desc.round(4).to_markdown(), encoding="utf-8")

    adf = adf_table(prices, ret)
    adf.to_csv(OUT_TABLES / "table02_adf.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table02_adf.md").write_text(adf.round(4).to_markdown(index=False), encoding="utf-8")

    dep = serial_dependence(ret)
    dep.to_csv(OUT_TABLES / "table03_serial_dependence.csv", encoding="utf-8-sig")
    (OUT_TABLES / "table03_serial_dependence.md").write_text(dep.round(4).to_markdown(), encoding="utf-8")

    panel_series(prices, OUT_FIGURES / "fig01_prices.png", "Index level (points)", "A-share broad indices: daily close, 2015-2024")
    panel_series(ret, OUT_FIGURES / "fig02_returns.png", "Log return (%)", "Daily log returns, 2015-2024")
    acf_figures(ret)

    print(desc.round(4).to_string())
    print("\n" + adf.round(4).to_string(index=False))
    print("\n" + dep.round(4).to_string())


if __name__ == "__main__":
    main()
