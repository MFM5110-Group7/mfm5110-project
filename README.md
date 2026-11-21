# A股宽基指数收益率与波动率时间序列研究（可复现管道）

本目录包含一项针对沪深300、创业板指、中证500、中证1000的日频收益率/波动率
建模与样本外预测研究。数据、代码、表格、图形和运行日志均可复现。

## 1. 环境

- Python 3.12.7（Anaconda）
- 依赖版本见 `requirements.txt`
- 主要库：`akshare==1.19.1`、`statsmodels==0.14.2`、`arch==9.0.0`、
  `numpy==1.26.4`、`pandas==2.2.2`、`scipy==1.13.1`、`matplotlib==3.9.2`

安装：

```bash
python -m pip install -r requirements.txt
# 若 arch 安装受网络/临时目录限制，也可将 wheel 解压到 .vendor，
# 代码会自动把 .vendor 加入 sys.path（见 scripts/04_garch.py）。
```

## 2. 数据

- 来源：AKShare `index_zh_a_hist`（东方财富日频指数行情）
- 底层接口：`https://push2his.eastmoney.com/api/qt/stock/kline/get`，
  `fqt=0`（不复权）
- 接口：`ak.index_zh_a_hist(symbol=code, period="daily",
  start_date="20150101", end_date="20241231")`
- 指数代码：`000300` 沪深300、`399006` 创业板指、`000905` 中证500、
  `000852` 中证1000
- 下载日期：2026-11-20（记录于 `data/processed/metadata.json`）
- 复权：指数收盘点位，无复权；不使用个股调整后收盘价
- 字段：日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、
  涨跌额、换手率；建模仅用 `date` 与 `close`
- 对齐：四只指数交易日取交集（2431 个交易日完全一致）
- 缺失/停牌：原始数据无缺失收盘、无重复日期、无零成交量；指数无停牌
- 异常值：保留（2015 股灾、2020-02-03、2024-09/10 行情均保留）；
  稳健性中仅用 5 日滚动平方收益代理，不做缩尾或删除

原始文件：`data/raw/<code>.csv`；处理后：
`data/processed/prices_wide.csv`、`returns_wide.csv`、`metadata.json`。

收益率定义：

```
r_t = 100 * [ln(P_t) - ln(P_{t-1})]
```

## 3. 样本划分与预测规则

- 全样本：2015-01-06 至 2024-12-31，共 2430 个收益率观测
- 训练集：前 80%，1944 个观测（2015-01-06 至 2022-12-28）
- 测试集：后 20%，486 个观测（2022-12-29 至 2024-12-31）
- 所有模型使用完全相同的测试日期
- 滚动一步向前预测，扩展窗口；预测 t 日仅使用 t-1 及之前的信息
- ARMA：每 10 个交易日重新 MLE 估计参数；两次重估之间用最后一次参数
  对新增观测做滤波（`append(refit=False)`），预测仍为一步向前
- GARCH 族：每个预测起点都重新估计参数（以上一次估计为初值）
- 参数是否重估、波动率代理定义均在报告中明确说明

## 4. 目录结构

```
.
├─ data/
│  ├─ raw/                     # AKShare 原始日线（每指数一个 csv）
│  └─ processed/               # 对齐后的价格、收益率、元数据
├─ src/
│  ├─ config.py                # 路径、样本、模型网格等配置
│  └─ plots.py                 # 统一绘图样式
├─ scripts/
│  ├─ 01_download.py           # 下载 + 清洗 + 对齐 + 收益率
│  ├─ 02_descriptives.py       # 描述统计、ADF、LB、ARCH-LM、ACF/PACF
│  ├─ 03_arma.py               # ARMA 6x6 定阶、诊断、滚动预测
│  ├─ 04_garch.py              # GARCH/EGARCH/GJR × normal/t、滚动预测
│  ├─ 05_evaluate.py           # MAE/RMSE/QLIKE、DM、图表、汇总
│  └─ 06_stage_report.py       # 从表格自动生成 STAGE_REPORT.md
├─ outputs/
│  ├─ tables/                  # table00–table13（csv + md）
│  ├─ figures/                 # fig01–fig06
│  ├─ forecasts/               # arma_forecasts.csv, garch_forecasts.csv
│  ├─ logs/                    # 运行日志
│  └─ results_summary.json     # 关键结果索引
├─ STAGE_REPORT.md             # 第一阶段结果（自动生成，含表与图）
├─ run_all.py                  # 一键运行（可跳过已下载数据）
├─ requirements.txt
└─ README.md
```

## 5. 运行

```bash
# 全流程（若 data/raw 已存在则跳过下载）
python run_all.py

# 强制重新下载
python run_all.py --redownload

# 或分步运行
python scripts/01_download.py
python scripts/02_descriptives.py
python scripts/03_arma.py
python scripts/04_garch.py
python scripts/05_evaluate.py
python scripts/06_stage_report.py
```

预计运行时间（单机、单线程）：下载约 1.5 分钟；描述统计约 10 秒；
ARMA 定阶 + 滚动约 20–30 分钟；GARCH 族约 10–15 分钟；评价约 5 秒。
实际时间取决于 CPU 与网络。

## 6. 输出文件与预期结构

### 表格

| 文件 | 内容 |
|---|---|
| `table00_sample_split.csv` | 样本区间、训练/测试观测数 |
| `table01_descriptives.csv` | N、均值、标准差、偏度、超额峰度、JB、最小/最大 |
| `table02_adf.csv` | 价格/收益率 ADF（常数、常数+趋势）、滞后、统计量、p |
| `table03_serial_dependence.csv` | 收益率/平方收益率 LB(10,20)、ARCH-LM(10) |
| `table04_arma_grid.csv` | 4 指数 × 36 个 ARMA(p,q) 的 AIC/BIC/收敛 |
| `table05_arma_selection.csv` | AIC/BIC 最优阶数、最终选择、样本区间、对数似然 |
| `table06_arma_params.csv` | 选定 ARMA 的系数、标准误、t、p、置信区间 |
| `table07_arma_diagnostics.csv` | ARMA 残差 LB、平方残差 LB、ARCH-LM |
| `table08_garch_comparison.csv` | 6 个 GARCH 模型的 llf/AIC/BIC/持续性/诊断 |
| `table09_garch_params.csv` | GARCH 族参数、标准误、t、p、显著性 |
| `table10_garch_diagnostics.csv` | 标准化残差 LB、平方 LB、ARCH-LM |
| `table11_return_forecast_eval.csv` | 收益率预测 MAE/RMSE/MSE（分指数与 Pooled） |
| `table12_vol_forecast_eval.csv` | 方差/波动率预测 MAE/RMSE/QLIKE（r²、5 日代理） |
| `table13_dm_tests.csv` | 收益率与波动率预测的 Diebold-Mariano 检验 |

### 图形

- `fig01_prices.png`：四指数收盘点位
- `fig02_returns.png`：四指数日对数收益率
- `fig03_acf_pacf_<code>.png`：收益率/平方收益率 ACF、PACF
- `fig04_return_forecasts.png`：测试集实际收益率 vs ARMA/零预测
- `fig05_volatility_forecasts.png`：测试集 |r|、5 日实现波动 vs GARCH 族 σ̂
- `fig06_volatility_rmse.png`：GARCH 族方差预测 RMSE 柱状图

### 预测明细

- `outputs/forecasts/arma_forecasts.csv`：
  `date,index,actual,yhat_arma,n_obs,refit`
- `outputs/forecasts/garch_forecasts.csv`：
  `date,index,model,dist,actual,mu,h,sigma,n_obs,fit_ok,convergence_flag`

### 阶段报告

- `STAGE_REPORT.md`：由 `scripts/06_stage_report.py` 从上述表格自动生成，
  包含数据、检验、模型比较、样本外评价、图表、局限与下一轮建议。
- `outputs/results_summary.json`：关键结果索引（样本划分、ARMA 阶数、
  最优模型、GARCH AIC/BIC 选择）。

## 7. 方法要点

- ARMA 定阶：p,q=0..5，`trend='c'`，AIC/BIC 均记录；**只在收敛模型中比较**
  （中证1000 的高阶 ARMA 存在不收敛，非收敛模型的 AIC/BIC 不参与选择）；
  AIC 用于最终一步预测模型（BIC 作为稳健性/简洁性对照，若不一致在报告中说明）。
- GARCH 族：GARCH(1,1)、GJR-GARCH(1,1)、EGARCH(1,1) × normal/t，
  常数均值；按 AIC/BIC 与残差诊断比较；样本外用 BIC 选定的分布。
- 波动率代理：主指标 `r_t^2`；稳健性 `r2_5(t)=mean(r_{t-4..t}^2)`；
  另报告 σ̂ 对 |r_t| 的 MAE/RMSE。r² 作为代理噪声很大，结论需谨慎。
- DM 检验：Newey-West HAC（lag=5），仅作辅助，不替代 MAE/RMSE 主表。

## 8. 复现注意事项

- `arch==9.0.0` 的 `ARCHModel.fit` 使用 `starting_values` 而非旧版的
  `start_params`；代码已按 9.0.0 编写。
- 若本机无法联网，`data/raw` 与 `data/processed` 已包含本次运行的数据，
  可直接从 `scripts/02_descriptives.py` 开始复现；下载脚本会记录失败信息。
- 所有表格数字由脚本从数据重新计算，未手工填入。
