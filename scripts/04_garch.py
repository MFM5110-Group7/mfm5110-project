"""GARCH-family volatility models.

Models
------
GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1), each with Gaussian and Student-t
innovations, constant mean equation.

Estimation
----------
* Training sample: full 6-model comparison by log-likelihood, AIC, BIC and
  residual diagnostics.
* Out-of-sample: expanding window, one-step-ahead conditional variance.
  Parameters are re-estimated at every forecast origin; the previous estimate
  is used as the starting value.  A forecast for day t uses returns up to t-1.

Outputs
-------
outputs/tables/table08_garch_comparison.csv
outputs/tables/table09_garch_params.csv
outputs/tables/table10_garch_diagnostics.csv
outputs/forecasts/garch_forecasts.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vendor"))
from src.config import DATA_PROCESSED, OUT_FORECASTS, OUT_TABLES, TRAIN_FRAC  # noqa: E402

from arch import arch_model  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402

warnings.filterwarnings("ignore")

SPECS = {
    "GARCH": dict(vol="GARCH", p=1, q=1),
    "GJR-GARCH": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1),
}
DISTS = ("normal", "t")


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fit_model(y: np.ndarray, spec: str, dist: str, start_params=None):
    try:
        kwargs = dict(mean="Constant", dist=dist, **SPECS[spec])
        am = arch_model(y, **kwargs)
        kw = {"disp": "off", "show_warning": False}
        if start_params is not None:
            kw["starting_values"] = np.asarray(start_params, dtype=float)
        res = am.fit(**kw)
        if not np.isfinite(res.loglikelihood):
            return None
        return res
    except Exception:
        return None


def persistence(spec: str, params: pd.Series) -> float:
    a = float(params.get("alpha[1]", np.nan))
    b = float(params.get("beta[1]", np.nan))
    g = float(params.get("gamma[1]", 0.0))
    if spec == "GARCH":
        return a + b
    if spec == "GJR-GARCH":
        return a + b + 0.5 * g
    return b  # EGARCH: coefficient on log h_{t-1}


def diag(res) -> dict:
    z = np.asarray(res.std_resid, dtype=float)
    z = z[np.isfinite(z)]
    lb = acorr_ljungbox(z, lags=[10, 20], return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=[10, 20], return_df=True)
    lm_stat, lm_p, _, _ = het_arch(z, nlags=10)
    return {
        "LB_z_Q10": float(lb.loc[10, "lb_stat"]), "LB_z_p10": float(lb.loc[10, "lb_pvalue"]),
        "LB_z_Q20": float(lb.loc[20, "lb_stat"]), "LB_z_p20": float(lb.loc[20, "lb_pvalue"]),
        "LB_z2_Q10": float(lb2.loc[10, "lb_stat"]), "LB_z2_p10": float(lb2.loc[10, "lb_pvalue"]),
        "LB_z2_Q20": float(lb2.loc[20, "lb_stat"]), "LB_z2_p20": float(lb2.loc[20, "lb_pvalue"]),
        "ARCHLM10_stat": float(lm_stat), "ARCHLM10_p": float(lm_p),
    }


def rolling_garch(y: np.ndarray, n_train: int, spec: str, dist: str) -> tuple[pd.DataFrame, dict]:
    n = len(y)
    res = fit_model(y[:n_train], spec, dist)
    if res is None:
        raise RuntimeError(f"initial {spec}-{dist} fit failed")
    prev_params = np.asarray(res.params, dtype=float)
    rows = []
    n_fail = 0
    for i in range(n_train, n):
        ok = True
        if i > n_train:
            new = fit_model(y[:i], spec, dist, start_params=prev_params)
            if new is not None:
                res = new
                prev_params = np.asarray(new.params, dtype=float)
            else:
                ok = False
                n_fail += 1
        if ok:
            try:
                f = res.forecast(horizon=1, reindex=False)
                h = float(f.variance.values[-1, 0])
                mu = float(f.mean.values[-1, 0])
            except Exception:
                ok = False
                n_fail += 1
        if not ok or not np.isfinite(h) or h <= 0:
            h = float(np.var(y[:i], ddof=1))
            mu = float(np.mean(y[:i]))
            ok = False
        rows.append({"i": i, "mu": mu, "h": h, "sigma": np.sqrt(h),
                     "n_obs": i, "fit_ok": bool(ok), "convergence_flag": int(getattr(res, "convergence_flag", -1))})
    return pd.DataFrame(rows), {"n_fail": n_fail}


def main() -> None:
    ret = pd.read_csv(DATA_PROCESSED / "returns_wide.csv", parse_dates=["date"], index_col="date")
    n = len(ret)
    n_train = int(np.floor(TRAIN_FRAC * n))

    comp_rows, param_rows, diag_rows = [], [], []
    fc_frames = []
    for code in ret.columns:
        y = ret[code].to_numpy(dtype=float)
        y_train = y[:n_train]
        print(f"[GARCH] {code}: training comparison ...", flush=True)
        results: dict[tuple[str, str], object] = {}
        for spec in SPECS:
            for dist in DISTS:
                t0 = time.time()
                res = fit_model(y_train, spec, dist)
                if res is None:
                    print(f"  {spec}-{dist}: FAILED", flush=True)
                    continue
                results[(spec, dist)] = res
                p = res.params
                d = diag(res)
                row = {
                    "index": code, "model": spec, "dist": dist,
                    "loglik": float(res.loglikelihood), "aic": float(res.aic), "bic": float(res.bic),
                    "n_params": int(len(p)), "convergence_flag": int(getattr(res, "convergence_flag", -1)),
                    "alpha": float(p.get("alpha[1]", np.nan)),
                    "gamma": float(p.get("gamma[1]", np.nan)),
                    "beta": float(p.get("beta[1]", np.nan)),
                    "nu": float(p.get("nu", np.nan)),
                    "persistence": persistence(spec, p),
                }
                row.update(d)
                comp_rows.append(row)
                for name in p.index:
                    param_rows.append({
                        "index": code, "model": spec, "dist": dist, "param": name,
                        "coef": float(p[name]), "std_err": float(res.std_err[name]),
                        "t": float(res.tvalues[name]), "p_value": float(res.pvalues[name]),
                        "sig": stars(float(res.pvalues[name])),
                    })
                diag_rows.append({"index": code, "model": spec, "dist": dist, **d})
                print(f"  {spec}-{dist}: llf={res.loglikelihood:.1f} aic={res.aic:.1f} "
                      f"bic={res.bic:.1f} ({time.time() - t0:.2f}s)", flush=True)

        comp = pd.DataFrame([r for r in comp_rows if r["index"] == code])
        comp["aic_rank"] = comp["aic"].rank(method="first")
        comp["bic_rank"] = comp["bic"].rank(method="first")

        # Distribution choice for each volatility specification: lower BIC.
        chosen_dist = {}
        for spec in SPECS:
            sub = comp[comp["model"] == spec]
            if sub.empty:
                continue
            chosen_dist[spec] = sub.sort_values("bic").iloc[0]["dist"]
        print(f"[GARCH] {code}: chosen dist by BIC = {chosen_dist}", flush=True)

        for spec, dist in chosen_dist.items():
            print(f"[GARCH] {code}: rolling {spec}-{dist} over {n - n_train} days ...", flush=True)
            t0 = time.time()
            fc, info = rolling_garch(y, n_train, spec, dist)
            fc["date"] = ret.index[fc["i"]]
            fc["index"] = code
            fc["model"] = spec
            fc["dist"] = dist
            fc["actual"] = y[fc["i"].to_numpy()]
            fc_frames.append(fc[["date", "index", "model", "dist", "actual", "mu", "h", "sigma",
                                 "n_obs", "fit_ok", "convergence_flag"]])
            print(f"           done in {time.time() - t0:.1f}s, failures={info['n_fail']}", flush=True)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_TABLES / "table08_garch_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table08_garch_comparison.md").write_text(comp_df.round(4).to_markdown(index=False), encoding="utf-8")
    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(OUT_TABLES / "table09_garch_params.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table09_garch_params.md").write_text(params_df.round(4).to_markdown(index=False), encoding="utf-8")
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_TABLES / "table10_garch_diagnostics.csv", index=False, encoding="utf-8-sig")
    (OUT_TABLES / "table10_garch_diagnostics.md").write_text(diag_df.round(4).to_markdown(index=False), encoding="utf-8")
    pd.concat(fc_frames, ignore_index=True).to_csv(OUT_FORECASTS / "garch_forecasts.csv", index=False, encoding="utf-8-sig")
    print("[GARCH] all outputs written", flush=True)


if __name__ == "__main__":
    main()

