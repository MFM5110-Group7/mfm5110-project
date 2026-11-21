"""Data download and cleaning.

Source
------
AKShare 1.19.1, function ``index_zh_a_hist`` (Eastmoney daily index quotes).
The raw payload is saved unchanged (apart from column renaming) under
``data/raw/<code>.csv``; the processed wide tables are built from those files.

Reproduce
---------
    python scripts/01_download.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    DATA_PROCESSED,
    DATA_RAW,
    DOWNLOAD_DATE,
    INDICES,
    RET_SCALE,
    SAMPLE_END,
    SAMPLE_START,
)

COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover",
}


def fetch_one(code: str, retries: int = 3) -> pd.DataFrame:
    import akshare as ak

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.index_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=SAMPLE_START.replace("-", ""),
                end_date=SAMPLE_END.replace("-", ""),
            )
            if df is None or df.empty:
                raise RuntimeError(f"empty payload for {code}")
            return df
        except Exception as exc:  # network hiccup -> retry
            last_err = exc
            print(f"[warn] {code} attempt {attempt}/{retries} failed: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {code}: {last_err}")


def clean_one(df: pd.DataFrame, code: str) -> tuple[pd.DataFrame, dict]:
    df = df.rename(columns=COLUMN_MAP).copy()
    required = ["date", "open", "close", "high", "low", "volume", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{code}: missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    df = df[(df["date"] >= pd.Timestamp(SAMPLE_START)) & (df["date"] <= pd.Timestamp(SAMPLE_END))]
    df = df[["date", "open", "close", "high", "low", "volume", "amount",
             "amplitude", "pct_chg", "change", "turnover"]]

    qc = {
        "code": code,
        "rows": int(len(df)),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_close": int(df["close"].isna().sum()),
        "nonpositive_close": int((df["close"] <= 0).sum()),
        "zero_volume_days": int((df["volume"] == 0).sum()),
        "max_abs_pct_chg": float(df["pct_chg"].abs().max()),
    }
    return df, qc


def main() -> None:
    all_frames: dict[str, pd.DataFrame] = {}
    qc_rows: list[dict] = []

    for code, meta in INDICES.items():
        print(f"[download] {code} {meta['name_en']}")
        raw = fetch_one(code)
        clean, qc = clean_one(raw, code)
        clean.to_csv(DATA_RAW / f"{code}.csv", index=False, encoding="utf-8-sig")
        all_frames[code] = clean
        qc_rows.append(qc)
        print(f"           rows={qc['rows']} {qc['start']} -> {qc['end']}")

    # Align on the intersection of trading dates.  Index calendars are nearly
    # identical; any date not common to all four is dropped and counted.
    closes = pd.DataFrame({code: df.set_index("date")["close"] for code, df in all_frames.items()})
    n_union = len(closes)
    closes = closes.dropna(how="any").sort_index()
    n_common = len(closes)

    prices = closes.reset_index().rename(columns={"index": "date"})
    prices.to_csv(DATA_PROCESSED / "prices_wide.csv", index=False, encoding="utf-8-sig")

    returns = RET_SCALE * np.log(prices.set_index("date")).diff()
    returns = returns.dropna(how="any")
    returns.index.name = "date"
    returns.to_csv(DATA_PROCESSED / "returns_wide.csv", encoding="utf-8-sig")

    metadata = {
        "source": "AKShare index_zh_a_hist (Eastmoney daily index quotes)",
        "akshare_version": __import__("akshare").__version__,
        "api_call": "ak.index_zh_a_hist(symbol=code, period='daily', "
                    f"start_date='{SAMPLE_START.replace('-', '')}', "
                    f"end_date='{SAMPLE_END.replace('-', '')}')",
        "source_endpoint": "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        "source_doc": "https://akshare.akfamily.xyz/data/index/index.html",
        "download_date": DOWNLOAD_DATE,
        "download_timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_requested": [SAMPLE_START, SAMPLE_END],
        "fields_raw_cn": list(COLUMN_MAP.keys()),
        "fields_used": ["date", "close"],
        "adjustment": "none - fqt=0; index closing level (not a dividend-adjusted stock price)",
        "return_formula": "r_t = 100 * (ln(P_t) - ln(P_{t-1}))",
        "alignment": "inner join on trading dates common to all four indices",
        "n_union_dates": int(n_union),
        "n_common_dates": int(n_common),
        "n_returns": int(len(returns)),
        "quality_checks": qc_rows,
    }
    with open(DATA_PROCESSED / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(f"[done] prices {prices.shape}, returns {returns.shape}, common dates {n_common}/{n_union}")


if __name__ == "__main__":
    main()
