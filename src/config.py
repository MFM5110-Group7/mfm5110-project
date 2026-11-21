"""Project-wide configuration.

All paths are resolved relative to the repository root so the pipeline can be
run from any working directory.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_LOGS = ROOT / "outputs" / "logs"
OUT_FORECASTS = ROOT / "outputs" / "forecasts"

for _p in (DATA_RAW, DATA_PROCESSED, OUT_TABLES, OUT_FIGURES, OUT_LOGS, OUT_FORECASTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample and data source
# ---------------------------------------------------------------------------
SAMPLE_START = "2015-01-01"
SAMPLE_END = "2024-12-31"
DOWNLOAD_DATE = datetime.now().strftime("%Y-%m-%d")   # recorded in metadata.json

# AKShare `index_zh_a_hist` symbols (Eastmoney index codes).
INDICES: dict[str, dict[str, str]] = {
    "000300": {"name_cn": "沪深300",   "name_en": "CSI 300",   "exchange": "SH"},
    "399006": {"name_cn": "创业板指",  "name_en": "ChiNext",   "exchange": "SZ"},
    "000905": {"name_cn": "中证500",   "name_en": "CSI 500",   "exchange": "SH"},
    "000852": {"name_cn": "中证1000",  "name_en": "CSI 1000",  "exchange": "SH"},
}

TRAIN_FRAC = 0.80          # chronological 80/20 split
RET_SCALE = 100.0          # r_t = 100 * [ln(P_t) - ln(P_{t-1})]

# ARMA grid
ARMA_MAX_P = 5
ARMA_MAX_Q = 5

# GARCH settings
GARCH_DISTS = ("normal", "t")
GARCH_SPECS = ("GARCH", "EGARCH", "GJR-GARCH")

# Rolling-forecast settings
REFIT_EVERY = 1            # 1 = re-estimate parameters at every forecast origin
MIN_TRAIN_OBS = 500        # safety guard

# Numerical / reporting
SEED = 20241120
FIG_DPI = 160
