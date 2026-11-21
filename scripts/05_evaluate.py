"""Out-of-sample evaluation of return and volatility forecasts.

All models are compared on the identical chronological test set (last 20%),
using one-step-ahead forecasts from expanding windows.

Return metrics: MAE, RMSE, MSE (returns in percent).
Volatility metrics: variance forecasts h_hat against r_t^2 (primary proxy),
and against a 5-day trailing mean of r_t^2 (robustness), plus QLIKE.
Diebold-Mariano tests use Newey-West HAC standard errors.

Outputs
-------
outputs/tables/table11_return_forecast_eval.csv
outputs/tables/table12_vol_forecast_eval.csv
outputs/tables/table13_dm_tests.csv
outputs/tables/table00_sample_split.csv
outputs/figures/fig04_return_forecasts.png
outputs/figures/fig05_volatility_forecasts.png
outputs/figures/fig06_volatility_rmse.png
outputs/results_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED, OUT_FIGURES, OUT_FORECASTS, OUT_TABLES, TRAIN_FRAC  # noqa: E402
from src.plots import COLORS, label, save, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

HAC_LAG = 5


# ---------------------------------------------------------------------------
# Metrics and tests
# ---------------------------------------------------------------------------
def err_metrics(y: np.ndarray, f: np.ndarray) -> dict:
    e = y - f
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MSE": float(np.mean(e ** 2)), "H": int(len(e))}


def qlike(y2: np.ndarray, h: np.ndarray) -> float:
    mask = (y2 > 0) & (h > 0) & np.isfinite(y2) & np.isfinite(h)
    ratio = y2[mask] / h[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def newey_west_var(d: np.ndarray, lag: int) -> float:
    d = d[np.isfinite(d)]
    n = len(d)
    mu = d.mean()
    var = np.mean((d - mu) ** 2)
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = np.mean((d[l:] - mu) * (d[:-l] - mu))
        var += 2.0 * w * cov
    return max(var, 1e-20)


def dm(y: np.ndarray, f1: np.ndarray, f2: np.ndarray, loss: str = "abs", lag: int = HAC_LAG) -> dict:
    e1, e2 = y - f1, y - f2
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1 ** 2 - e2 ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    mu = d.mean()
    stat = mu / np.sqrt(newey_west_var(d, lag) / n)
    p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return {"mean_loss_diff": float(mu), "DM_stat": float(stat), "DM_p": float(p), "H": int(n)}


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def main() -> None:
    ret = pd.read_csv(DATA_PROCESSED / "returns_wide.csv", parse_dates=["date"], index_col="date")
    n = len(ret)
    n_train = int(np.floor(TRAIN_FRAC * n))

    arma = pd.read_csv(OUT_FORECASTS / "arma_forecasts.csv", parse_dates=["date"], dtype={"index": str})
    garch = pd.read_csv(OUT_FORECASTS / "garch_forecasts.csv", parse_dates=["date"], dtype={"index": str})

    # Constant-mean and random-walk benchmarks on exactly the same dates.
    bench = []
    for code in ret.columns:
        y = ret[code].to_numpy(dtype=float)
        for i in range(n_train, n):
            bench.append({"date": ret.index[i], "index": code, "actual_bench": y[i],
                          "yhat_const": float(np.mean(y[:i])), "yhat_zero": 0.0})
    bench = pd.DataFrame(bench)
    ret_fc = arma.merge(bench, on=["date", "index"], how="left", validate="one_to_one")
    if not np.allclose(ret_fc["actual"], ret_fc["actual_bench"]):
        raise ValueError("actual return mismatch between ARMA forecasts and benchmark grid")

    # Sample split table.
    split_rows = []
    for code in ret.columns:
        split_rows.append({
            "index": code, "name": label(code), "n_prices": len(ret) + 1,
            "n_returns": len(ret), "n_train": n_train, "n_test": n - n_train,
            "train_start": str(ret.index[0].date()), "train_end": str(ret.index[n_train - 1].date()),
            "test_start": str(ret.index[n_train].date()), "test_end": str(ret.index[-1].date()),
        })
    split = pd.DataFrame(split_rows)
    split.to_csv(OUT_TABLES / "table00_sample_split.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table00_sample_split.md").write_text(split.to_markdown(index=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Return forecast evaluation
    # ------------------------------------------------------------------
    arma_orders = pd.read_csv(OUT_TABLES / "table05_arma_selection.csv", dtype={"index": str})
    order_map = {r["index"]: f"ARMA({int(r['chosen_p'])},{int(r['chosen_q'])})" for _, r in arma_orders.iterrows()}

    ret_rows = []
    for code in ret.columns:
        sub = ret_fc[ret_fc["index"] == code].sort_values("date")
        y = sub["actual"].to_numpy()
        models = {
            "Zero (RW)": sub["yhat_zero"].to_numpy(),
            "Constant mean": sub["yhat_const"].to_numpy(),
            order_map[code]: sub["yhat_arma"].to_numpy(),
        }
        for name, f in models.items():
            row = {"index": code, "model": name}
            row.update(err_metrics(y, f))
            ret_rows.append(row)
    ret_eval = pd.DataFrame(ret_rows)
    ret_eval["rank_MAE"] = ret_eval.groupby("index")["MAE"].rank(method="min")
    ret_eval["rank_RMSE"] = ret_eval.groupby("index")["RMSE"].rank(method="min")

    # Pooled (all index-day observations, equal weight per observation).
    pooled_rows = []
    for name, grp in ret_fc.melt(id_vars=["date", "index", "actual"],
                                 value_vars=["yhat_zero", "yhat_const", "yhat_arma"],
                                 var_name="model", value_name="yhat").groupby("model"):
        mapping = {"yhat_zero": "Zero (RW)", "yhat_const": "Constant mean", "yhat_arma": "ARMA"}
        row = {"index": "Pooled", "model": mapping[name]}
        row.update(err_metrics(grp["actual"].to_numpy(), grp["yhat"].to_numpy()))
        pooled_rows.append(row)
    # Rename pooled ARMA to a generic label (orders differ by index).
    pooled = pd.DataFrame(pooled_rows)
    pooled["model"] = pooled["model"].replace({"ARMA": "ARMA (index-specific order)"})
    pooled["rank_MAE"] = pooled["MAE"].rank(method="min")
    pooled["rank_RMSE"] = pooled["RMSE"].rank(method="min")
    ret_eval_full = pd.concat([ret_eval, pooled], ignore_index=True)
    ret_eval_full.to_csv(OUT_TABLES / "table11_return_forecast_eval.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table11_return_forecast_eval.md").write_text(
        ret_eval_full.round(4).to_markdown(index=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Volatility forecast evaluation
    # ------------------------------------------------------------------
    # Trailing 5-day mean of squared returns, observable at t.
    proxy5 = ret.pow(2).rolling(5, min_periods=5).mean()
    proxy5_long = proxy5.stack().rename("r2_5").reset_index()
    proxy5_long.columns = ["date", "index", "r2_5"]

    vol = garch.merge(proxy5_long, on=["date", "index"], how="left", validate="many_to_one")
    vol["r2"] = vol["actual"] ** 2
    vol["abs_r"] = vol["actual"].abs()
    vol["sigma_proxy5"] = np.sqrt(vol["r2_5"])

    # Constant-variance benchmark: expanding sample variance using y[:i].
    var_bench = []
    for code in ret.columns:
        y = ret[code].to_numpy(dtype=float)
        for i in range(n_train, n):
            var_bench.append({"date": ret.index[i], "index": code,
                              "h_const": float(np.var(y[:i], ddof=1))})
    var_bench = pd.DataFrame(var_bench)
    base_cols = vol[["date", "index", "actual", "r2", "abs_r", "r2_5", "sigma_proxy5"]].drop_duplicates(["date", "index"])
    cv = base_cols.merge(var_bench, on=["date", "index"], how="left", validate="one_to_one")
    cv["model"] = "Constant variance"
    cv["dist"] = "-"
    cv["h"] = cv["h_const"]
    cv["sigma"] = np.sqrt(cv["h_const"])
    cv["mu"] = np.nan
    cv["n_obs"] = n_train
    cv["fit_ok"] = True
    cv["convergence_flag"] = 0
    vol_all = pd.concat([vol, cv[vol.columns]], ignore_index=True)

    vol_rows = []
    for (code, model, dist), sub in vol_all.groupby(["index", "model", "dist"]):
        sub = sub.sort_values("date")
        y2 = sub["r2"].to_numpy()
        h = sub["h"].to_numpy()
        sig = sub["sigma"].to_numpy()
        row = {
            "index": code, "model": model, "dist": dist,
            "MAE_var": float(np.mean(np.abs(h - y2))),
            "RMSE_var": float(np.sqrt(np.mean((h - y2) ** 2))),
            "QLIKE": qlike(y2, h),
            "MAE_vol": float(np.mean(np.abs(sig - sub["abs_r"].to_numpy()))),
            "RMSE_vol": float(np.sqrt(np.mean((sig - sub["abs_r"].to_numpy()) ** 2))),
            "MAE_var5": float(np.mean(np.abs(h - sub["r2_5"].to_numpy()))),
            "RMSE_var5": float(np.sqrt(np.mean((h - sub["r2_5"].to_numpy()) ** 2))),
            "H": int(len(sub)),
        }
        vol_rows.append(row)
    vol_eval = pd.DataFrame(vol_rows)
    vol_eval["rank_RMSE_var"] = vol_eval.groupby("index")["RMSE_var"].rank(method="min")
    vol_eval["rank_QLIKE"] = vol_eval.groupby("index")["QLIKE"].rank(method="min")
    vol_eval["rank_RMSE_var5"] = vol_eval.groupby("index")["RMSE_var5"].rank(method="min")

    # Pooled volatility metrics (equal weight per index-day).
    pooled_vol = []
    for (model, dist), sub in vol_all.groupby(["model", "dist"]):
        row = {"index": "Pooled", "model": model, "dist": dist,
               "MAE_var": float(np.mean(np.abs(sub["h"] - sub["r2"]))),
               "RMSE_var": float(np.sqrt(np.mean((sub["h"] - sub["r2"]) ** 2))),
               "QLIKE": qlike(sub["r2"].to_numpy(), sub["h"].to_numpy()),
               "MAE_vol": float(np.mean(np.abs(sub["sigma"] - sub["abs_r"]))),
               "RMSE_vol": float(np.sqrt(np.mean((sub["sigma"] - sub["abs_r"]) ** 2))),
               "MAE_var5": float(np.mean(np.abs(sub["h"] - sub["r2_5"]))),
               "RMSE_var5": float(np.sqrt(np.mean((sub["h"] - sub["r2_5"]) ** 2))),
               "H": int(len(sub))}
        pooled_vol.append(row)
    pooled_vol = pd.DataFrame(pooled_vol)
    pooled_vol["rank_RMSE_var"] = pooled_vol["RMSE_var"].rank(method="min")
    pooled_vol["rank_QLIKE"] = pooled_vol["QLIKE"].rank(method="min")
    pooled_vol["rank_RMSE_var5"] = pooled_vol["RMSE_var5"].rank(method="min")
    vol_eval_full = pd.concat([vol_eval, pooled_vol], ignore_index=True)
    vol_eval_full.to_csv(OUT_TABLES / "table12_vol_forecast_eval.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table12_vol_forecast_eval.md").write_text(
        vol_eval_full.round(4).to_markdown(index=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Diebold-Mariano tests
    # ------------------------------------------------------------------
    dm_rows = []
    for code in ret.columns:
        sub = ret_fc[ret_fc["index"] == code].sort_values("date")
        y = sub["actual"].to_numpy()
        pairs = [
            (order_map[code], "Zero (RW)", sub["yhat_arma"].to_numpy(), sub["yhat_zero"].to_numpy()),
            (order_map[code], "Constant mean", sub["yhat_arma"].to_numpy(), sub["yhat_const"].to_numpy()),
            ("Constant mean", "Zero (RW)", sub["yhat_const"].to_numpy(), sub["yhat_zero"].to_numpy()),
        ]
        for m1, m2, f1, f2 in pairs:
            for loss in ("abs", "sq"):
                r = dm(y, f1, f2, loss=loss)
                dm_rows.append({"index": code, "market": "return", "model_1": m1, "model_2": m2,
                                "loss": loss, **r, "p_sig": stars(r["DM_p"])})

    for code in ret.columns:
        sub = vol_all[vol_all["index"] == code]
        pivot_h = sub.pivot_table(index="date", columns="model", values="h")
        pivot_y = sub.groupby("date")["r2"].first()
        pairs = [("GARCH", "EGARCH"), ("GARCH", "GJR-GARCH"), ("EGARCH", "GJR-GARCH"),
                 ("GARCH", "Constant variance"), ("EGARCH", "Constant variance"),
                 ("GJR-GARCH", "Constant variance")]
        for m1, m2 in pairs:
            if m1 not in pivot_h or m2 not in pivot_h:
                continue
            for loss in ("abs", "sq"):
                r = dm(pivot_y.to_numpy(), pivot_h[m1].to_numpy(), pivot_h[m2].to_numpy(), loss=loss)
                dm_rows.append({"index": code, "market": "variance", "model_1": m1, "model_2": m2,
                                "loss": loss, **r, "p_sig": stars(r["DM_p"])})
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(OUT_TABLES / "table13_dm_tests.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table13_dm_tests.md").write_text(dm_df.round(4).to_markdown(index=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True)
    for ax, code in zip(axes.ravel(), ret.columns):
        sub = ret_fc[ret_fc["index"] == code].sort_values("date")
        ax.plot(sub["date"], sub["actual"], color="0.65", lw=0.6, label="Actual")
        ax.plot(sub["date"], sub["yhat_arma"], color=COLORS.get(code, "C3"), lw=0.9, label=order_map[code])
        ax.axhline(0.0, color="black", lw=0.6, ls="--", label="Zero / RW")
        ax.set_title(label(code))
        ax.set_ylabel("Return (%)")
        if code == list(ret.columns)[0]:
            ax.legend(loc="lower left", ncol=2, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
            lab.set_ha("right")
    fig.suptitle("Test-set return forecasts (one-step-ahead, expanding window)", fontsize=12, y=0.995)
    save(fig, OUT_FIGURES / "fig04_return_forecasts.png", tight=False)

    # Volatility: model conditional sigma vs |r| and 5-day proxy.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True)
    for ax, code in zip(axes.ravel(), ret.columns):
        sub = vol[vol["index"] == code]
        first = sub[sub["model"] == "GARCH"].sort_values("date")
        ax.plot(first["date"], first["abs_r"], color="0.75", lw=0.55, label="|r_t|")
        ax.plot(first["date"], first["sigma_proxy5"], color="black", lw=0.9, ls="--", label="5-day realized")
        for model, color in zip(["GARCH", "EGARCH", "GJR-GARCH"], ["#1f77b4", "#d62728", "#2ca02c"]):
            s = sub[sub["model"] == model].sort_values("date")
            ax.plot(s["date"], s["sigma"], color=color, lw=0.9, label=model)
        ax.set_title(label(code))
        ax.set_ylabel("Volatility (% per day)")
        if code == list(ret.columns)[0]:
            ax.legend(loc="upper left", ncol=2, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
            lab.set_ha("right")
    fig.suptitle("Test-set conditional volatility forecasts", fontsize=12, y=0.995)
    save(fig, OUT_FIGURES / "fig05_volatility_forecasts.png", tight=False)

    # Grouped bar chart of RMSE (variance units).
    per_index = vol_eval.copy()
    models = ["GARCH", "EGARCH", "GJR-GARCH"]
    x = np.arange(len(ret.columns))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for j, model in enumerate(models):
        vals = [per_index[(per_index["index"] == c) & (per_index["model"] == model)]["RMSE_var"].iloc[0]
                for c in ret.columns]
        ax.bar(x + (j - 1) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([label(c) for c in ret.columns], rotation=0, fontsize=8)
    ax.set_ylabel("RMSE (variance units)")
    ax.set_title("Volatility forecast RMSE against r_t^2")
    ax.legend()
    save(fig, OUT_FIGURES / "fig06_volatility_rmse.png")

    # ------------------------------------------------------------------
    # Compact JSON for the report
    # ------------------------------------------------------------------
    best_ret = {}
    for code in list(ret.columns) + ["Pooled"]:
        sub = ret_eval_full[ret_eval_full["index"] == code].sort_values("RMSE")
        best_ret[code] = {"best_by_RMSE": sub.iloc[0]["model"], "RMSE": float(sub.iloc[0]["RMSE"]),
                          "MAE": float(sub.iloc[0]["MAE"])}
    best_vol = {}
    for code in list(ret.columns) + ["Pooled"]:
        sub = vol_eval_full[vol_eval_full["index"] == code].sort_values("RMSE_var")
        best_vol[code] = {"best_by_RMSE_var": sub.iloc[0]["model"], "dist": sub.iloc[0]["dist"],
                          "RMSE_var": float(sub.iloc[0]["RMSE_var"]), "QLIKE": float(sub.iloc[0]["QLIKE"])}
    summary = {
        "n_returns": n, "n_train": n_train, "n_test": n - n_train,
        "train": [str(ret.index[0].date()), str(ret.index[n_train - 1].date())],
        "test": [str(ret.index[n_train].date()), str(ret.index[-1].date())],
        "arma_orders": order_map,
        "best_return_model": best_ret,
        "best_vol_model": best_vol,
        "garch_comparison_best": {},
    }
    comp = pd.read_csv(OUT_TABLES / "table08_garch_comparison.csv", dtype={"index": str})
    for code, sub in comp.groupby("index"):
        a = sub.sort_values("aic").iloc[0]
        b = sub.sort_values("bic").iloc[0]
        summary["garch_comparison_best"][code] = {
            "AIC": f"{a['model']}-{a['dist']} ({a['aic']:.2f})",
            "BIC": f"{b['model']}-{b['dist']} ({b['bic']:.2f})",
        }
    (OUT_FORECASTS.parent / "results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(ret_eval_full.round(4).to_string(index=False))
    print("\n" + vol_eval_full.round(4).to_string(index=False))
    print("\n" + dm_df.round(4).to_string(index=False))
    print("\n[summary]", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
