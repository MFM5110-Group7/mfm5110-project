"""Shared plotting helpers (English labels to avoid CJK font issues)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIG_DPI, INDICES  # noqa: E402

COLORS = {
    "000300": "#1f4e79",
    "399006": "#c00000",
    "000905": "#2e7d32",
    "000852": "#7b1fa2",
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.bbox": "tight",
    })


def label(code: str) -> str:
    return f"{INDICES[code]['name_en']} ({code})"


def save(fig: plt.Figure, path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def panel_series(df: pd.DataFrame, path, ylabel: str, title: str) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    for ax, code in zip(axes.ravel(), df.columns):
        ax.plot(df.index, df[code], color=COLORS.get(code, "black"), linewidth=0.8)
        ax.set_title(label(code))
        ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("Date")
    fig.suptitle(title, fontsize=12, y=0.995)
    save(fig, path, tight=False)
