"""ARMA mean models: order selection, estimation, diagnostics and rolling forecasts.

Design
------
* Order grid p,q = 0..5 on the chronological training sample (first 80%).
* AIC and BIC are both recorded.  AIC is used for the out-of-sample model
  because the target is one-step-ahead forecasting; BIC is reported as the
  parsimony-based alternative.
* Out-of-sample: expanding window, one-step-ahead.  Parameters are re-estimated
  every ``ARMA_REFIT_EVERY`` trading days (default 10); between refits the
  latest observations are filtered with the last estimated parameters.  Every
  forecast for day t uses information up to t-1 only.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    ARMA_MAX_P,
    ARMA_MAX_Q,
    DATA_PROCESSED,
    OUT_FORECASTS,
    OUT_TABLES,
    TRAIN_FRAC,
)

warnings.filterwarnings("ignore")
ARMA_REFIT_EVERY = 10
RNG_SEED = 20241120


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def safe_fit(y: np.ndarray, order: tuple[int, int, int], start_params=None):
    try:
        kwargs = {}
        if start_params is not None:
            kwargs["start_params"] = np.asarray(start_params, dtype=float)
        res = ARIMA(y, order=order, trend="c").fit(**kwargs)
        return res
    except Exception:
        return None


def select_orders(y_train: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    for p in range(ARMA_MAX_P + 1):
        for q in range(ARMA_MAX_Q + 1):
            res = safe_fit(y_train, (p, 0, q))
            conv = bool(getattr(res, "mle_retvals", {}).get("converged", False)) if res is not None else False
            rows.append({
                "p": p,
                "q": q,
                "aic": float(res.aic) if res is not None else np.nan,
                "bic": float(res.bic) if res is not None else np.nan,
                "llf": float(res.llf) if res is not None else np.nan,
                "converged": conv,
                "n_params": (p + q + 2) if res is not None else np.nan,
            })
    grid = pd.DataFrame(rows)
    valid_all = grid.dropna(subset=["aic", "bic"])
    if valid_all.empty:
        raise RuntimeError("all ARMA fits failed")
    # Only converged models are eligible: AIC/BIC of a non-converged fit are
    # not reliable.  If (unexpectedly) none converged, fall back to all fits.
    valid = valid_all[valid_all["converged"]]
    if valid.empty:
        valid = valid_all
    best_aic = valid.sort_values(["aic", "p", "q"]).iloc[0]
    best_bic = valid.sort_values(["bic", "p", "q"]).iloc[0]
    chosen = best_aic if not (best_aic["p"] == best_bic["p"] and best_aic["q"] == best_bic["q"]) else best_aic
    info = {
        "best_aic_p": int(best_aic["p"]), "best_aic_q": int(best_aic["q"]),
        "best_aic": float(best_aic["aic"]),
        "best_bic_p": int(best_bic["p"]), "best_bic_q": int(best_bic["q"]),
        "best_bic": float(best_bic["bic"]),
        "chosen_p": int(chosen["p"]), "chosen_q": int(chosen["q"]),
        "criterion_used": "AIC",
        "aic_bic_agree": bool(best_aic["p"] == best_bic["p"] and best_aic["q"] == best_bic["q"]),
        "n_converged": int(valid_all["converged"].sum()),
        "n_grid": int(len(valid_all)),
    }
    return grid, info


def param_frame(res) -> pd.DataFrame:
    names = list(res.model.param_names)
    coef = np.asarray(res.params, dtype=float).ravel()
    se = np.asarray(res.bse, dtype=float).ravel()
    tvals = np.asarray(res.tvalues, dtype=float).ravel()
    pvals = np.asarray(res.pvalues, dtype=float).ravel()
    ci = np.asarray(res.conf_int(), dtype=float)
    out = pd.DataFrame({
        "param": names,
        "coef": coef,
        "std_err": se,
        "t": tvals,
        "p_value": pvals,
        "ci_low": ci[:, 0],
        "ci_high": ci[:, 1],
    })
    out["sig"] = out["p_value"].map(stars)
    return out


def diagnostics(res) -> dict:
    resid = np.asarray(res.resid, dtype=float)
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    lb2 = acorr_ljungbox(resid ** 2, lags=[10, 20], return_df=True)
    lm_stat, lm_p, lm_f, lm_fp = het_arch(resid, nlags=10)
    return {
        "LB_resid_Q10": float(lb.loc[10, "lb_stat"]), "LB_resid_p10": float(lb.loc[10, "lb_pvalue"]),
        "LB_resid_Q20": float(lb.loc[20, "lb_stat"]), "LB_resid_p20": float(lb.loc[20, "lb_pvalue"]),
        "LB_resid2_Q10": float(lb2.loc[10, "lb_stat"]), "LB_resid2_p10": float(lb2.loc[10, "lb_pvalue"]),
        "LB_resid2_Q20": float(lb2.loc[20, "lb_stat"]), "LB_resid2_p20": float(lb2.loc[20, "lb_pvalue"]),
        "ARCHLM10_stat": float(lm_stat), "ARCHLM10_p": float(lm_p),
    }


def rolling_forecast(y: np.ndarray, n_train: int, order: tuple[int, int, int]) -> tuple[pd.DataFrame, list[dict]]:
    n = len(y)
    res = safe_fit(y[:n_train], order)
    if res is None:
        raise RuntimeError(f"initial ARMA{order} fit failed")
    preds, events = [], []
    for i in range(n_train, n):
        is_refit = ((i - n_train) % ARMA_REFIT_EVERY == 0)
        if is_refit and i > n_train:
            new_res = safe_fit(y[:i], order, start_params=res.params)
            if new_res is not None:
                res = new_res
                events.append({"date_idx": i, "event": "refit_ok"})
            else:
                events.append({"date_idx": i, "event": "refit_failed_keep_old"})
        try:
            fc = float(np.ravel(res.forecast(1))[0])
        except Exception:
            fc = float(np.mean(y[:i]))
            events.append({"date_idx": i, "event": "forecast_failed_mean_fallback"})
        preds.append({"i": i, "yhat": fc, "n_obs": i, "refit": is_refit})
        # Extend the information set by the realised observation.
        try:
            res = res.append(np.array([y[i]]), refit=False)
        except Exception:
            res = safe_fit(y[: i + 1], order)
    return pd.DataFrame(preds), events


def main() -> None:
    ret = pd.read_csv(DATA_PROCESSED / "returns_wide.csv", parse_dates=["date"], index_col="date")
    n = len(ret)
    n_train = int(np.floor(TRAIN_FRAC * n))
    rng = np.random.default_rng(RNG_SEED)

    grid_all, select_rows, param_rows, diag_rows, fc_frames = [], [], [], [], []
    for code in ret.columns:
        t0 = time.time()
        y = ret[code].to_numpy(dtype=float)
        y_train = y[:n_train]
        print(f"[ARMA] {code}: grid {ARMA_MAX_P + 1}x{ARMA_MAX_Q + 1} on {n_train} obs ...", flush=True)
        grid, info = select_orders(y_train)
        grid.insert(0, "index", code)
        grid_all.append(grid)

        p, q = info["chosen_p"], info["chosen_q"]
        res = safe_fit(y_train, (p, 0, q))
        if res is None:
            raise RuntimeError(f"{code}: chosen ARMA({p},{q}) failed")
        info.update({"index": code, "n_train": n_train, "n_test": n - n_train,
                     "train_start": str(ret.index[0].date()), "train_end": str(ret.index[n_train - 1].date()),
                     "test_start": str(ret.index[n_train].date()), "test_end": str(ret.index[-1].date()),
                     "loglik": float(res.llf), "aic": float(res.aic), "bic": float(res.bic),
                     "params": list(res.model.param_names)})
        select_rows.append(info)

        pf = param_frame(res)
        pf.insert(0, "index", code)
        pf.insert(1, "p", p)
        pf.insert(2, "q", q)
        param_rows.append(pf)

        dg = diagnostics(res)
        dg.update({"index": code, "p": p, "q": q})
        diag_rows.append(dg)

        print(f"[ARMA] {code}: AIC=({info['best_aic_p']},{info['best_aic_q']}) "
              f"BIC=({info['best_bic_p']},{info['best_bic_q']}) -> using ({p},{q}); "
              f"rolling {n - n_train} steps ...", flush=True)
        fc, events = rolling_forecast(y, n_train, (p, 0, q))
        fc["date"] = ret.index[fc["i"]]
        fc["index"] = code
        fc["actual"] = y[fc["i"].to_numpy()]
        fc["yhat_arma"] = fc["yhat"]
        fc_frames.append(fc[["date", "index", "actual", "yhat_arma", "n_obs", "refit"]])
        print(f"[ARMA] {code}: done in {time.time() - t0:.1f}s; refit failures="
              f"{sum(e['event'] != 'refit_ok' for e in events)}", flush=True)

    pd.concat(grid_all, ignore_index=True).to_csv(OUT_TABLES / "table04_arma_grid.csv", index=False, encoding="utf-8-sig")
    sel = pd.DataFrame(select_rows)
    sel.to_csv(OUT_TABLES / "table05_arma_selection.csv", index=False, encoding="utf-8-sig")
    pd.concat(param_rows, ignore_index=True).to_csv(OUT_TABLES / "table06_arma_params.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diag_rows).to_csv(OUT_TABLES / "table07_arma_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.concat(fc_frames, ignore_index=True).to_csv(OUT_FORECASTS / "arma_forecasts.csv", index=False, encoding="utf-8-sig")

    for name, df in [("table04_arma_grid", pd.concat(grid_all, ignore_index=True)),
                     ("table05_arma_selection", sel),
                     ("table06_arma_params", pd.concat(param_rows, ignore_index=True)),
                     ("table07_arma_diagnostics", pd.DataFrame(diag_rows))]:
        (OUT_TABLES / f"{name}.md").write_text(df.round(4).to_markdown(index=False), encoding="utf-8")
    print("[ARMA] all outputs written", flush=True)


if __name__ == "__main__":
    main()
