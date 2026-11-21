"""Run the whole reproducible pipeline.

Usage
-----
    python run_all.py                 # reuse existing raw data if present
    python run_all.py --redownload    # force a fresh download
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ("01_download.py", "download and clean data"),
    ("02_descriptives.py", "descriptives and tests"),
    ("03_arma.py", "ARMA selection and rolling forecasts"),
    ("04_garch.py", "GARCH family and rolling forecasts"),
    ("05_evaluate.py", "forecast evaluation and figures"),
    ("06_stage_report.py", "assemble the stage report"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redownload", action="store_true")
    args = parser.parse_args()

    (ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    for i, (script, desc) in enumerate(SCRIPTS):
        if script == "01_download.py" and not args.redownload:
            raw = list((ROOT / "data" / "raw").glob("*.csv"))
            processed = ROOT / "data" / "processed" / "returns_wide.csv"
            if len(raw) >= 4 and processed.exists():
                print(f"[skip] raw/processed data already present ({len(raw)} files); use --redownload to refresh")
                continue
        print(f"\n=== [{i + 1}/{len(SCRIPTS)}] {script}: {desc} ===", flush=True)
        t0 = time.time()
        log_path = ROOT / "outputs" / "logs" / f"{Path(script).stem}.log"
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace",
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            print(tail)
            raise SystemExit(f"{script} failed with exit code {proc.returncode}; see {log_path}")
        print(f"    done in {time.time() - t0:.1f}s; log={log_path}")
    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
