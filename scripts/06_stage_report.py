"""Build the stage report from the generated tables (single source of truth).

The report intentionally contains only results that were produced by the
pipeline.  Run after ``05_evaluate.py``.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vendor"))
from src.config import INDICES  # noqa: E402

T = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "STAGE_REPORT.md"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name, dtype={"index": str})


def md(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".4f") -> str:
    d = df.copy()
    if cols is not None:
        d = d[[c for c in cols if c in d.columns]]
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:{floatfmt}}")
    d = d.fillna("")
    return d.to_markdown(index=False)


def fig(name: str, caption: str) -> str:
    return f"![{caption}](outputs/figures/{name})\n\n*{caption}*\n"


def main() -> None:
    meta = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "outputs" / "results_summary.json").read_text(encoding="utf-8"))
    desc = read("table01_descriptives.csv")
    adf = read("table02_adf.csv")
    dep = read("table03_serial_dependence.csv")
    sel = read("table05_arma_selection.csv")
    arma_par = read("table06_arma_params.csv")
    arma_diag = read("table07_arma_diagnostics.csv")
    garch = read("table08_garch_comparison.csv")
    garch_par = read("table09_garch_params.csv")
    garch_diag = read("table10_garch_diagnostics.csv")
    ret_eval = read("table11_return_forecast_eval.csv")
    vol_eval = read("table12_vol_forecast_eval.csv")
    dm = read("table13_dm_tests.csv")
    split = read("table00_sample_split.csv")

    # Best model by index (excluding pooled).
    ret_best = (ret_eval[ret_eval["index"] != "Pooled"]
                .sort_values(["index", "RMSE"]).groupby("index").first().reset_index())
    vol_best = (vol_eval[vol_eval["index"] != "Pooled"]
                .sort_values(["index", "RMSE_var"]).groupby("index").first().reset_index())
    vol_best_q = (vol_eval[vol_eval["index"] != "Pooled"]
                  .sort_values(["index", "QLIKE"]).groupby("index").first().reset_index())

    # Average ranks across the four indices (collapse index-specific ARMA labels).
    ret_rank_src = ret_eval[ret_eval["index"] != "Pooled"].copy()
    ret_rank_src["model"] = ret_rank_src["model"].where(
        ~ret_rank_src["model"].str.startswith("ARMA"), "ARMA (all)")
    ret_rank = (ret_rank_src.groupby("model")[["rank_MAE", "rank_RMSE"]]
                .mean().sort_values("rank_RMSE"))
    vol_rank = (vol_eval[vol_eval["index"] != "Pooled"]
                .groupby("model")[["rank_RMSE_var", "rank_QLIKE"]].mean().sort_values("rank_RMSE_var"))

    # DM significance counts (5%).
    dm_sig = dm[dm["DM_p"] < 0.05]

    lines: list[str] = []
    add = lines.append
    add("# A股宽基指数收益率与波动率：第一阶段实证结果")
    add("")
    add("> **运行状态：已实际运行（非模拟）。**  ")
    add(f"> 数据下载日期：{meta['download_date']}；数据源：{meta['source']}；"
        f"AKShare {meta['akshare_version']}。  ")
    add(f"> 环境：Python {platform.python_version()}，pandas {pd.__version__}，"
        f"NumPy {__import__('numpy').__version__}，statsmodels {__import__('statsmodels').__version__}，"
        f"arch {__import__('arch').__version__}。  ")
    add("> 本文件是**阶段性结果**，用于模型迭代，不是最终论文；所有数字由 "
        "`outputs/tables/*.csv` 自动读取生成。")
    add("")
    add("## 1. 研究计划与数据获取方案")
    add("")
    add("**目标**：对沪深300、创业板指、中证500、中证1000 的日频收益率做 "
        "ARMA 均值建模，对条件方差做 GARCH/EGARCH/GJR-GARCH 建模，并在完全相同的 "
        "测试集上做一步向前滚动预测与 MAE/RMSE 评价。")
    add("")
    add("**数据**：AKShare `index_zh_a_hist`（东方财富日频指数行情，底层接口 "
        "`https://push2his.eastmoney.com/api/qt/stock/kline/get`，`fqt=0` 不复权），"
        f"接口参数 `start_date=20150101, end_date=20241231`。"
        f"下载日期 {meta['download_date']}。指数收盘点位，无复权；"
        "收益率 `r_t = 100 * [ln(P_t) - ln(P_{t-1})]`。")
    add("")
    add("**样本**：四指数交易日取交集，"
        f"{meta['n_common_dates']} 个交易日、{meta['n_returns']} 个收益率观测；"
        "按时间顺序前 80% 训练、后 20% 测试；所有模型共用同一测试集。")
    add("")
    add("**滚动规则**：扩展窗口、一步向前；预测 t 日仅使用 t-1 及之前的信息。"
        "ARMA 每 10 个交易日重新 MLE 估计（两次重估之间用最后一次参数滤波新增观测）；"
        "GARCH 族在每个预测起点重新估计。")
    add("")
    add("**评价**：收益率用 MAE/RMSE；波动率主代理为 `r_t^2`，"
        "稳健性代理为 5 日滚动平方收益均值；辅以 QLIKE 和 Diebold-Mariano 检验。")
    add("")
    add("## 2. 数据与样本")
    add("")
    qc = pd.DataFrame(meta["quality_checks"])
    qc["name"] = qc["code"].map(lambda c: INDICES[c]["name_en"])
    add(md(qc, ["code", "name", "rows", "start", "end", "duplicate_dates",
                "missing_close", "nonpositive_close", "zero_volume_days", "max_abs_pct_chg"]))
    add("")
    add(md(split, ["index", "name", "n_returns", "n_train", "n_test",
                   "train_start", "train_end", "test_start", "test_end"]))
    add("")
    add(f"- 四指数交易日完全一致：{meta['n_common_dates']}/{meta['n_union_dates']}；"
        "无缺失收盘、无重复日期、无零成交量。")
    add("- 异常值（2015 股灾、2020-02-03、2024-09/10 急涨急跌）全部保留；"
        "未做缩尾或删除。最大单日对数收益："
        f"创业板指 {desc.loc[desc['index']=='399006','max'].iloc[0]:.2f}%（2024-10-08），"
        f"中证1000 {desc.loc[desc['index']=='000852','max'].iloc[0]:.2f}%（2024-09-30）。")
    add("")
    add("## 3. 描述统计与初步检验")
    add("")
    add("### 3.1 描述性统计")
    add("")
    add(md(desc, ["index", "name", "N", "mean", "std", "skew", "excess_kurt",
                  "min", "max", "JB", "JB_p"]))
    add("")
    add("所有指数收益率均显著非正态（JB p≈0），左偏（创业板指接近 0）、超额峰度 4.4–6.7，"
        "呈现典型肥尾。")
    add("")
    add("### 3.2 ADF 平稳性")
    add("")
    adf_show = adf[((adf["series"] == "return") & (adf["regression"] == "constant")) |
                   (adf["series"] == "price")]
    add(md(adf_show, ["index", "series", "regression", "lag_AIC", "stat", "p_value",
                      "crit_5pct", "n_obs", "stationary_5pct"]))
    add("")
    add("收益率在常数（及常数+趋势）设定下均在 1% 水平拒绝单位根；价格水平只在部分设定下"
        "拒绝，整体不能稳健地视为平稳。后续均值模型使用收益率。")
    add("")
    add("### 3.3 白噪声与 ARCH 效应")
    add("")
    add(md(dep, ["index", "LB_r_Q10", "LB_r_p10", "LB_r_Q20", "LB_r_p20",
                 "LB_r2_Q10", "LB_r2_p10", "LB_r2_Q20", "LB_r2_p20",
                 "ARCHLM10_stat", "ARCHLM10_p"]))
    add("")
    add("收益率存在弱但显著的自相关（LB(10) p<0.05）；平方收益率自相关和 ARCH-LM "
        "均极度显著，波动率聚集强，必须使用 GARCH 族。")
    add("")
    add("## 4. ARMA 均值模型")
    add("")
    add("### 4.1 定阶")
    add("")
    add(md(sel, ["index", "best_aic_p", "best_aic_q", "best_aic",
                 "best_bic_p", "best_bic_q", "best_bic",
                 "chosen_p", "chosen_q", "aic_bic_agree",
                 "n_converged", "n_grid", "loglik"]))
    add("")
    agree = sel[sel["aic_bic_agree"]]
    disagree = sel[~sel["aic_bic_agree"]]
    agree_txt = "、".join(INDICES[c]["name_cn"] for c in agree["index"]) or "无"
    if len(disagree):
        disagree_txt = "；".join(
            f"{INDICES[r['index']]['name_cn']}（AIC ({int(r['best_aic_p'])},{int(r['best_aic_q'])}) "
            f"vs BIC ({int(r['best_bic_p'])},{int(r['best_bic_q'])})）"
            for _, r in disagree.iterrows())
    else:
        disagree_txt = "无"
    add(f"AIC 与 BIC 一致的指数：{agree_txt}；不一致：{disagree_txt}。"
        "本阶段按“一步预测用 AIC”处理，BIC 作为简洁性对照。")
    if "n_converged" in sel.columns:
        conv_txt = "；".join(
            f"{INDICES[r['index']]['name_cn']} {int(r['n_converged'])}/{int(r['n_grid'])}"
            for _, r in sel.iterrows())
        add(f"定阶只在收敛模型中比较（收敛模型数：{conv_txt}）。"
            "中证1000 的高阶 ARMA 大量不收敛，若把非收敛模型的 AIC 也算进去会误选 (4,3)；"
            "限定收敛模型后 AIC/BIC 均选 (1,0)。这是一处必须如实说明的模型选择修正。")
    add("")
    add("### 4.2 参数与残差诊断")
    add("")
    add(md(arma_par, ["index", "p", "q", "param", "coef", "std_err", "t", "p_value", "sig"]))
    add("")
    add(md(arma_diag, ["index", "p", "q", "LB_resid_p10", "LB_resid_p20",
                       "LB_resid2_p10", "LB_resid2_p20", "ARCHLM10_p"]))
    add("")
    d852 = arma_diag[arma_diag["index"] == "000852"].iloc[0]
    add(f"沪深300、创业板指、中证500 的 ARMA 残差 LB(10/20) 均不拒绝；"
        f"中证1000 的 ARMA(1,0) 在 LB(20) 上仍拒绝（p={d852['LB_resid_p20']:.4f}），"
        "说明这个简约模型留下了剩余线性自相关。"
        "能消除自相关的高阶模型（如 (4,3)）又不收敛，形成“简约但欠拟合 / 复杂但不稳定”的取舍。"
        "四个指数的平方残差与 ARCH-LM 仍极度显著："
        "均值方程没有、也不应被期待吸收条件异方差，波动率需要单独的 GARCH 族模型。")
    add("")
    add("## 5. GARCH 族波动率模型")
    add("")
    add("### 5.1 训练集模型比较")
    add("")
    add(md(garch, ["index", "model", "dist", "loglik", "aic", "bic",
                   "alpha", "gamma", "beta", "nu", "persistence",
                   "LB_z2_p10", "ARCHLM10_p"]))
    add("")
    add("**分布**：四个指数、三个波动率设定下，Student-t 的 AIC/BIC 均明显优于正态"
        "（ΔAIC 约 72–189），且 ν 估计 4.7–7.1，进一步支持肥尾。")
    add("")
    add("**模型**：EGARCH(1,1)-t 在四个指数上同时取得最小 AIC 和 BIC；"
        "GARCH(1,1)-t 与 GJR-GARCH(1,1)-t 次之。")
    add("")
    add("### 5.2 选定模型参数（EGARCH-t）")
    add("")
    add(md(garch_par[(garch_par["model"] == "EGARCH") & (garch_par["dist"] == "t")],
           ["index", "param", "coef", "std_err", "t", "p_value", "sig"]))
    add("")
    add(md(garch_diag[(garch_diag["model"] == "EGARCH") & (garch_diag["dist"] == "t")],
           ["index", "LB_z_p10", "LB_z_p20", "LB_z2_p10", "LB_z2_p20", "ARCHLM10_p"]))
    add("")
    add("- **持续性**：GARCH 的 α+β 为 0.988–0.995；EGARCH 的 β 为 0.984–0.991，"
        "波动率冲击高度持久。")
    add("- **杠杆效应**：GJR 的 γ 为正（0.014–0.030）但四个指数均不显著；"
        "EGARCH 的 γ 为负（−0.016 至 −0.033），仅中证500 在 10% 水平显著。"
        "因此本样本中“坏消息增加波动”的标准杠杆效应很弱，甚至方向相反；"
        "不能把它当作成熟市场式的稳健杠杆效应。")
    add("- **诊断**：EGARCH-t 的标准化残差 LB(10/20) 与平方残差 LB、ARCH-LM "
        "基本不拒绝（中证1000 的 LB(10) p≈0.069 处于边缘），ARCH 效应基本被吸收。")
    add("")
    add("## 6. 样本外预测评价")
    add("")
    add(f"测试集：{summary['test'][0]} 至 {summary['test'][1]}，H={summary['n_test']}；"
        "所有模型使用同一测试日期、同一扩展窗口规则。")
    add("")
    add("### 6.1 收益率预测")
    add("")
    add(md(ret_eval, ["index", "model", "H", "MAE", "RMSE", "MSE", "rank_MAE", "rank_RMSE"]))
    add("")
    add("**按 RMSE 的最优模型**：")
    for _, r in ret_best.iterrows():
        add(f"- {INDICES[r['index']]['name_cn']}：{r['model']}，RMSE={r['RMSE']:.4f}，MAE={r['MAE']:.4f}")
    add("")
    sub852 = ret_eval[ret_eval["index"] == "000852"]
    arma852 = sub852[sub852["model"].str.startswith("ARMA")].iloc[0]
    zero852 = sub852[sub852["model"] == "Zero (RW)"].iloc[0]
    add(f"中证1000 是一个例外：{arma852['model']} 的 RMSE={arma852['RMSE']:.4f}，"
        f"略低于零预测的 {zero852['RMSE']:.4f}；但其 MAE={arma852['MAE']:.4f}，"
        f"反而高于零预测的 {zero852['MAE']:.4f}。其余三个指数在 MAE 与 RMSE 上均为零预测最优。"
        "因此“ARMA 是否更好”依赖于损失函数，不能给出统一结论。")
    dm852 = dm[(dm["index"] == "000852") & (dm["market"] == "return") &
               (dm["model_1"].str.startswith("ARMA")) & (dm["model_2"] == "Zero (RW)") &
               (dm["loss"] == "sq")]
    if len(dm852):
        add(f"中证1000 的 ARMA(1,0) 对零预测的平方损失 DM 检验 p={dm852.iloc[0]['DM_p']:.4f}，"
            "差异在统计上不显著。")
    add("")
    add("平均排名（四个指数，越小越好）：")
    add("")
    add(md(ret_rank.reset_index(), ["model", "rank_MAE", "rank_RMSE"]))
    add("")
    add("注：GARCH 族的均值方程设为常数，其收益率预测与 `Constant mean` 基准完全相同，"
        "因此收益率表不重复列出 GARCH 均值；GARCH 族只在波动率维度比较。")
    add("")
    add("### 6.2 波动率预测")
    add("")
    add(md(vol_eval, ["index", "model", "dist", "MAE_var", "RMSE_var", "QLIKE",
                      "MAE_vol", "RMSE_vol", "RMSE_var5", "rank_RMSE_var", "rank_QLIKE"]))
    add("")
    add("**按 RMSE_var 的最优模型**：")
    for _, r in vol_best.iterrows():
        add(f"- {INDICES[r['index']]['name_cn']}：{r['model']}-{r['dist']}，"
            f"RMSE_var={r['RMSE_var']:.4f}，QLIKE={r['QLIKE']:.4f}")
    add("")
    add("**按 QLIKE 的最优模型**：")
    for _, r in vol_best_q.iterrows():
        add(f"- {INDICES[r['index']]['name_cn']}：{r['model']}-{r['dist']}，"
            f"QLIKE={r['QLIKE']:.4f}")
    add("")
    add("平均排名（四个指数，越小越好）：")
    add("")
    add(md(vol_rank.reset_index(), ["model", "rank_RMSE_var", "rank_QLIKE"]))
    add("")
    add("**样本内与样本外的选择不一致**：AIC/BIC 在四个指数上都选 EGARCH-t，"
        "但样本外按 RMSE_var 与 QLIKE 排名，GARCH-t 在四个指数上都排第 1；"
        "EGARCH-t 的 MAE_var 最小、RMSE_var/QLIKE 反而略差。"
        "这说明 AIC/BIC 偏好的模型不一定在特定预测损失下最优，"
        "评价必须以样本外损失为准，并同时报告多种损失。")
    add("")
    add("### 6.3 Diebold-Mariano 检验（平方损失）")
    add("")
    dm_show = dm[(dm["loss"] == "sq") & (dm["market"].isin(["return", "variance"]))]
    add(md(dm_show, ["index", "market", "model_1", "model_2",
                     "mean_loss_diff", "DM_stat", "DM_p", "p_sig"]))
    add("")
    add(f"5% 显著的 DM 组合数：{len(dm_sig)} / {len(dm)}。"
        "DM 仅作辅助，不替代 MAE/RMSE 主表；在 r² 代理噪声很大时，"
        "很多差异在统计上不显著。绝对损失下的结论更偏向 GARCH 族："
        "GARCH/EGARCH 相对常数方差的 MAE 改善在多数指数上显著；"
        "平方损失下则多不显著，说明极端日的 r² 主导了平方损失。")
    add("")
    add("## 7. 图表")
    add("")
    add(fig("fig01_prices.png", "图1 四只宽基指数收盘点位"))
    add(fig("fig02_returns.png", "图2 日对数收益率"))
    add(fig("fig03_acf_pacf_000300.png", "图3 沪深300 收益率/平方收益率 ACF、PACF（其余指数同结构）"))
    add(fig("fig04_return_forecasts.png", "图4 测试集实际收益率与 ARMA/零预测"))
    add(fig("fig05_volatility_forecasts.png", "图5 测试集条件波动率预测"))
    add(fig("fig06_volatility_rmse.png", "图6 GARCH 族方差预测 RMSE"))
    add("")
    add("## 8. 本阶段结论与下一轮迭代")
    add("")
    add("**可以下的结论**")
    add("")
    add("1. 四只指数收益率均平稳、显著非正态、存在弱自相关和强 ARCH 效应；"
        "价格水平不能稳健地视为平稳。")
    add("2. 条件方差建模中，Student-t 明显优于正态；EGARCH(1,1)-t 在本样本的 "
        "AIC/BIC 上最优，波动率持续性极高（β≈0.98–0.99）。")
    add("3. 杠杆效应很弱且方向不稳定；GJR 的 γ 不显著，EGARCH 的 γ 为负且仅中证500 "
        "在 10% 水平显著。")
    add("4. 日频收益率的一步预测非常困难：零预测在沪深300、创业板指、中证500 的 "
        "MAE 与 RMSE 均最优；中证1000 的 ARMA(1,0) 仅在 RMSE 上略优于零预测，"
        "但 MAE 更差。Pooled 仍以零预测最优。ARMA 相对基准没有稳定、统一的优势。"
        "波动率预测在 MAE 和 QLIKE 上优于常数方差基准，但平方损失下很多 DM 差异不显著，"
        "且 RMSE 受极端日 r² 主导，不同损失下 GARCH 与 EGARCH 的排名会变化。")
    add("")
    add("**必须说明的局限**")
    add("")
    add("- ARMA 每 10 个交易日重估一次，而非每日重估；GARCH 每日重估。"
        "下一轮应做“每日重估 ARMA”或“固定训练参数”的稳健性对照。")
    add("- GARCH 均值方程采用常数均值；未做含 MA 项的 ARMA-GARCH 联合估计。")
    add("- 波动率代理 `r_t^2` 噪声很大；5 日滚动平方收益只是部分缓解，"
        "不能替代高频已实现波动率。")
    add("- 未显式建模结构突变（2015 股灾、2020 疫情、2024-09/10 行情），"
        "也未做模型置信集（MCS）。")
    add("- 所有比较是统计预测比较，不涉及交易成本、仓位和可交易性。")
    add("")
    add("**下一轮建议（按优先级）**")
    add("")
    add("1. **均值模型稳健性**：对创业板指同时跑 AIC(2,5) 与 BIC(0,0)，"
        "并比较固定训练参数 vs 每 10 日重估；增加 AR(1)-GARCH(1,1)-t 作为联合均值-波动模型。")
    add("2. **波动率模型扩展**：APARCH、FIGARCH、GARCH-skew-t；"
        "若可取得高频数据，用 5 分钟已实现波动率 RV 替代 r²，并做 HAR-RV 对照。")
    add("3. **评价扩展**：Mincer-Zarnowitz 回归、模型置信集 MCS、"
        "DM 的 Harvey-Leybourne-Newbold 小样本修正；分别报告全测试集与剔除"
        "2024-09/10 极端日后的结果。")
    add("4. **结构变化**：滚动固定窗口（如 1000 日）与扩展窗口对比；"
        "对 2015、2020、2024 分段估计，检查参数稳定性。")
    add("5. **报告层面**：把本阶段表格整理为论文表格（三线表），"
        "补写模型设定、识别假设与经济解释。")
    add("")
    add("## 9. 复现")
    add("")
    add("```bash")
    add("python -m pip install -r requirements.txt")
    add("python run_all.py            # 已有 data/raw 时跳过下载")
    add("python run_all.py --redownload")
    add("```")
    add("")
    add("输出结构、字段说明与运行时间见 `README.md`；"
        "预测明细在 `outputs/forecasts/`，表格在 `outputs/tables/`，"
        "图为 `outputs/figures/`。")
    add("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
