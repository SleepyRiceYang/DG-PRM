# Rollout 分析程序说明 (analysis_rollout)

本程序对**对话式说服 (Persuasion) 的 Rollout 轨迹数据**进行多维度分析，生成结构化报告与可视化图表，用于评估模型表现、人格/决策风格差异、动态时序与策略选择等。

---

## 一、分析入口与整体输出

本仓库现在提供 **三类分析入口**，它们的输出都可以统一写入同一个 `<output_name>` 目录（推荐统一使用 `analysis_results`）：

- **main.py**：
  - 输入：单个 rollout JSON + persona JSON。
  - 作用：完整的用户级 / 人格级 / 策略 / 动态时序分析。
  - 输出：`<output_name>/reports/` 与 `<output_name>/figures/`（主体结构见第二节）。

- **unique_main.py**：
  - 输入：单个 rollout JSON（无需 persona）。
  - 作用：在同一实验内，对 **root 与各分支的 unique 段**（root 从 Round 1，branch 从 `branch_at_turn` 之后）做精细指标分析，关注干预后的真实变化。
  - 输出：写入同一 `<output_name>` 下的：
    - `reports/dynamics/unique_segments/`
    - `figures/dynamics/unique_segments/`

- **deep_main.py**：
  - 输入：同一目录下的多个 rollout JSON（不同 mode）。
  - 作用：跨模式（mode）比较整体表现、分支时机、root vs branch 胜率、用户在不同模式下的分支成功率等。
  - 输出：写入 `<output_name>/deep_analysis/`：
    - `deep_analysis/reports/`
    - `deep_analysis/figures/`

下文中的“输出目录结构”和后续小节，都是相对于某次运行指定的 `<output_name>`（默认 `analysis_results`）而言。

---

## 二、输出目录结构概览

一次完整分析（运行 main + unique + deep）后，输出目录大致结构如下（实际存在与否取决于是否运行对应入口以及数据是否支持该分析）：

| 类型 | 含义 | 格式 |
|------|------|------|
| **Reports（报告）** | 原始或聚合后的**数据表与统计结果**，供进一步计算、复现或导入其他工具使用。 | CSV（表格）、JSON（嵌套结构） |
| **Figures（图表）** | 基于上述数据绘制的**可视化图片与 PDF 报告**，用于直观查看趋势、对比和分布。 | PNG（位图）、PDF（多页报告） |

- **CSV**：行-列表格，可用 Excel /  pandas 打开，适合做表格式统计。
- **JSON**：键值结构，适合存储层级统计（如“按人格的指标汇总”）。
- **PNG**：单张高清图，适合汇报与论文插图。
- **PDF**：多页文档，用于“按人格/按用户”的逐页浏览（动态时序与个人层次）。

下文按**输出目录结构**列出所有文件，并逐一说明**每张图、每份报告的含义与作用**。



```
<output_name>/                        # 默认 analysis_results
├── reports/                          # 主流水线 main.py 的所有数据报告（CSV / JSON）
│   ├── 1_user_level_stats.csv
│   ├── 2_persona_level_stats.csv
│   ├── 3_global_stats.json
│   ├── critical_turns.csv
│   ├── critical_turns_summary.csv
│   ├── entropy_relief.csv
│   ├── dynamics/                    # 可选：dynamic/unique 等扩展分析的报告
│   │   └── unique_segments/         # unique_main.py 产生的 unique 段相关报告
│   └── strategy/                    # 策略相关报告（按类型分子目录）
│       ├── preference/              # 人格/风格对策略的偏好
│       ├── by_turn/                 # 按轮次的策略分布
│       ├── branch/                  # 分支重选策略
│       ├── overall/                 # 策略整体与多样性
│       ├── radar/                   # 雷达图用数据
│       └── outcome/                 # 策略与结果、转移矩阵
│
├── figures/                         # 主流水线 main.py 的所有可视化图表
│   ├── global_stats/                # 全局效果
│   ├── dynamics/                    # 动态时序与因果分支
│   │   ├── reports/                 # 动态时序 PDF 报告
│   │   │   ├── temporal_overall_level.pdf
│   │   │   ├── persona/             # 按人格、按决策风格的多页 PDF
│   │   │   └── individual/          # 按用户的个人层次 PDF
│   │   ├── unique_segments/         # unique_main.py 输出的 unique 段时序图
│   │   └── *.png                    # 其他动态相关图（临界轮次、熵缓解等）
│   └── strategy/                    # 策略分析（按类型分子目录）
│       ├── preference/
│       ├── by_turn/
│       ├── branch/
│       ├── overall/
│       ├── radar/
│       └── outcome/
│
└── deep_analysis/                   # deep_main.py 的多模式对比输出（可选）
    ├── reports/
    │   ├── combined_unique_trends.json
    │   ├── branch_timing_stats.json
    │   ├── mode_traj_stats.json
    │   └── mode_user_stats.json
    └── figures/
        ├── combine_metrics_dashboard.png
        ├── timing_success_fail_analysis.png
        └── mode_comprehensive_performance.png
```

---

## 三、Reports（报告）说明

### 3.1 根目录 reports/

| 文件名 | 含义与作用 |
|--------|-------------|
| **1_user_level_stats.csv** | 每个**用户**的汇总：总轨迹数、成功轨迹数、成功率、是否根轨迹成功、是否需要挽救、是否被挽救等。用于用户级效果评估与筛选。 |
| **2_persona_level_stats.csv** | 每个**人格×决策风格**组合的汇总：用户数、初始成功率、挽救率、轨迹成功率、人均成功轨迹数等。用于对比不同人格/风格的表现。 |
| **3_global_stats.json** | **全局**统计：总用户数、总轨迹数、成功轨迹数、全局初始成功率、挽救率、轨迹成功率。用于整体效果一目了然。 |
| **critical_turns.csv** | 每条**成功分支**对应的“分叉轮次”（BranchTurn + 1）。用于细查哪些轮次产生了成功挽救。 |
| **critical_turns_summary.csv** | **按轮次（1～10）**汇总：该轮总分支数、成功分支数、成功占比。用于与“临界轮次”图对应，看每轮分支质量。 |
| **entropy_relief.csv** | 父轨迹与子轨迹在分叉后**状态熵变化**（父熵 − 子熵）：正值表示分支带来更确定的状态。用于分析“挽救”是否伴随熵下降。 |

### 3.2 reports/strategy/ 下各子目录

- **preference/**  
  - `strategy_preference_bigfive.json/csv`：在**成功轨迹**中，每种 Big Five 人格下各策略的计数与占比，以及倾向强度（dominance、concentration）。  
  - `strategy_preference_style.json/csv`：同上，按**决策风格**统计。  
  → 用于回答“不同人格/风格是否倾向于不同策略集合、倾向程度如何”。

- **by_turn/**  
  - `strategy_by_turn.json`：每轮（Turn）的策略分布（计数与占比）及该轮主导策略。  
  - `strategy_by_turn_summary.csv`：每轮汇总（总轮数、主导策略、主导占比等）。  
  - `strategy_by_turn_counts.csv`：Turn × Strategy 的计数矩阵。  
  → 用于回答“不同 turn 是否有倾向的策略”。

- **branch/**  
  - `strategy_branch_reselection.json`：同一轮分支中，与父轨迹**策略相同/不同**的计数、总分支数、重选比例（re-selection sufficient ratio）及按轮汇总。  
  - `strategy_branch_crosstab.csv`：父策略 × 子策略的列联表（计数）。  
  - `strategy_branch_details.csv`：每条分支的父策略、子策略、是否相同等明细。  
  → 用于回答“同 turn 内重选策略的相关性与重选充分度”。

- **overall/**  
  - `strategy_overall.json`：全局策略计数、占比、top5、以及按 Outcome 的计数/占比。  
  - `strategy_overall_counts.csv`：各策略总使用次数。  
  - `strategy_diversity.csv`：每条轨迹的策略多样性（唯一策略数/轮数）。  
  → 用于策略总体统计与多样性评估。

- **radar/**  
  - `strategy_fingerprint.csv`：按 Big Five 与策略的**成功轨迹**内频率（用于人格雷达图）。  
  - `strategy_radar_success_vs_fail.json`：各策略在**成功/失败**轨迹中的占比列表，用于成功 vs 失败雷达图。  

- **outcome/**  
  - `strategy_success_rate.csv`：每个策略的“使用该策略的轮次中属于成功轨迹”的比例（turn 级成功率）。  
  - `strategy_transition_matrix.csv`：轨迹内**上一轮策略 → 下一轮策略**的转移计数矩阵。  
  → 用于策略与结果关系、策略转移模式。

---

## 四、Figures（图片与 PDF）说明

以下按**目录与文件名**逐项说明每张图/每个 PDF 的**定义**和**作用**，便于第一次接触的人快速理解。

### 4.1 figures/global_stats/（全局效果）

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **global_summary_bar.png** | 三根柱状图：初始成功率、挽救率、轨迹成功率；右上角文本框给出总用户数、总轨迹数、成功轨迹数。 | 一眼看清**整体表现**：根轨迹表现、挽救能力、轨迹级效率。 |
| **persona_matrix_dashboard.png** | 2×2 热力图矩阵：行=Big Five，列=决策风格；四个子图分别表示初始成功率、挽救率、轨迹成功率、人均成功轨迹数。 | 对比**不同人格×风格**在四项指标上的差异，发现优势/劣势组合。 |
| **user_trajectory_stats.png** | 横轴为用户，双轴图：柱状为每用户总轨迹数/成功轨迹数，折线为每用户成功轨迹占比。 | 查看**用户级**轨迹数量与成功率分布，识别高贡献或需关注用户。 |

### 4.2 figures/dynamics/（动态与因果）

> **Hs/Ha 两种视角说明**：本目录下有两类 Hs/Ha 时序图：
> - `temporal_overall_success_vs_fail.png`：按 Turn 聚合**整条轨迹**上的 Hs/Ha（root + 所有分支），展示整体成功 vs 失败轨迹群的动态演变。
> - `unique_segments/unique_metrics_dashboard.png`：仅统计 root 从 Round 1 开始、branch 从 `branch_at_turn` 之后的**unique 段**，并展示 Hs/Ha 的增量与 Z-score，更适合分析“干预后”的状态变化。
> 两者使用的是同一批 turn-level Hs/Ha 数据，但样本集合与聚合方式不同，因此数值曲线不完全一致：前者是“宏观气候图”，后者是“干预后的局部放大图”。

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **temporal_overall_success_vs_fail.png** | 2×2 子图：四个指标（状态熵、动作熵、认知失调、趋势动量）随 Turn 的变化；每条线为成功/失败轨迹群的均值，带标准差阴影。 | 从**整体**看成功与失败轨迹在**动态时序与状态演变**上的差异。 |
| **critical_turn_hist.png** | 横轴为轮次 1～10；每轮两根柱（总分支数、成功分支数），右轴为成功占比折线；柱上标注 N=总数，线上标注百分比。 | 看清**哪几轮**分支多、成功分支多、成功率高低，评估“临界挽救轮次”的分布。 |
| **entropy_relief_box.png** | 按分支结果（Success/Fail）分组的“熵缓解”（父熵−子熵）箱线图；零线参考。 | 看**成功分支**是否比失败分支带来更大的熵下降（更确定的后续状态）。 |

### 4.3 figures/dynamics/reports/（动态时序 PDF 报告）

| 文件名/目录 | 定义 | 作用 |
|-------------|------|------|
| **temporal_overall_level.pdf** | 单页：与 `temporal_overall_success_vs_fail.png` 内容一致的四指标 Success vs Fail 时序图。 | 提供与总体层次图一致的**可打印/分享**版本。 |
| **persona/temporal_persona_BigFive.pdf** | 多页：每页对应一种 Big Five 人格，页内为四指标 Success vs Fail 时序（该人格下所有风格合并）。 | 按**人格**翻页查看各人格的时序差异，人格与决策风格分开。 |
| **persona/temporal_persona_Style.pdf** | 多页：每页对应一种决策风格，页内为四指标 Success vs Fail 时序（该风格下所有人格合并）。 | 按**决策风格**翻页查看各风格的时序差异。 |
| **individual/temporal_individual_level.pdf** | 多页：每页若干用户；每用户左侧为信息卡（用户、人格、风格、总/成功轨迹数、成功率、初始成功、需挽救、是否被挽救），右侧为四指标在该用户成功/失败轨迹上的时序。 | 从**个人层次**查看每个用户的轨迹表现与四指标演变，便于个案分析。 |

### 4.4 figures/strategy/（策略分析，按子目录）

#### strategy/radar/

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **strategy_radar_fingerprint.png** | 雷达图：每条射线为一种策略，每条折线为一个 Big Five 人格在**成功轨迹**中的策略频率（闭合多边形）。 | 对比**不同人格**在成功轨迹上使用的**策略集合形状**，谁更偏哪些策略一目了然。 |
| **strategy_radar_success_vs_fail.png** | 同一批策略为轴；绿色多边形=成功轨迹中各策略占比，红色多边形=失败轨迹中各策略占比。 | **直接对比成功与失败轨迹**的策略集合差异，看清哪些策略更偏向成功/失败。 |

#### strategy/preference/

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **strategy_preference_bigfive.png** | 上方：Big Five × 策略的热力图（比例）；下方：每人格的倾向强度柱状图（Dominance / Concentration）。 | 看**人格**对策略的偏好与集中度，不混入决策风格。 |
| **strategy_preference_style.png** | 同上结构，行为**决策风格**、列为策略，下方为风格的倾向强度。 | 看**决策风格**对策略的偏好与集中度。 |

#### strategy/by_turn/

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **strategy_by_turn_heatmap.png** | 热力图：行=Turn，列=Strategy，颜色=该轮该策略的**比例**。 | 看**每一轮**各策略的占比，是否存在“某轮明显倾向某策略”。 |
| **strategy_by_turn_dominant.png** | 柱状图：横轴=Turn，纵轴=该轮**主导策略**的占比。 | 看每轮策略**集中程度**：柱子越高，该轮越集中在一个策略。 |

#### strategy/branch/

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **strategy_branch_reselection.png** | 左图：两根柱（与父策略相同 / 重选不同）+ 重选比例标注；右图：父策略×子策略的计数热力图。 | 看**同轮分支**中有多少选择与父轨迹不同策略、父-子策略的对应关系。 |
| **strategy_branch_reselection_by_turn.png** | 柱状图：横轴=轮次，纵轴=该轮分支的**重选比例**。 | 看**哪几轮**分支时更常“换策略”（重选充分度随轮次变化）。 |

#### strategy/overall/

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **strategy_overall_counts.png** | 单根柱状图：横轴=策略名，纵轴=该策略在**所有轮次**中的使用次数，柱顶标注数值。 | 看**策略总体使用量**排序，哪些策略最常用。 |
| **strategy_diversity.png** | 小提琴图：横轴=Big Five，纵轴=轨迹内“唯一策略数/轮数”（多样性得分），按成功/失败分色。 | 看**人格**下成功/失败轨迹的**策略多样性**差异。 |

#### strategy/outcome/

| 文件名 | 定义 | 作用 |
|--------|------|------|
| **strategy_success_rate.png** | 柱状图：横轴=策略，纵轴=使用该策略的轮次中属于**成功轨迹**的比例（turn 级成功率）。 | 看**各策略**与成功结果的关联强度。 |
| **strategy_transition_matrix.png** | 热力图：行=上一轮策略（From），列=下一轮策略（To），值为转移次数。 | 看**策略转移模式**：常从哪类策略转到哪类策略。 |

---

## 五、程序结构简介

本仓库为**单入口流水线**：读入 Rollout JSON 与用户人格 JSON，依次做全局评估、动态时序、因果分支、策略分析，并写入上述 reports 与 figures。

```
analysis_rollout/
├── README.md                 # 本说明
├── main.py                   # 入口：全局流水线分析
├── unique_main.py            # 入口：单模式 Unique Segment 深度分析
├── deep_main.py              # 入口：多模式跨文件对比深度分析
├── run.bash                  # 示例运行脚本
└── src/
    ├── data_loader.py        # 加载数据
    ├── metric_calculator.py  # 计算熵、增量、Z-score 等指标
    ├── visualizer.py         # 绘图类（含全局、动态、策略、Unique、Deep 对比图）
    └── analyzers/
        ├── global_evaluator.py
        ├── unique_segment_analyzer.py  # 新增：Unique 段提取与统计分析
        ├── mode_deep_analyzer.py      # 新增：多模式时机与效果对比分析
        ├── ... (其他分析器)
```

---

## 六、如何运行

### 6.1 全局综合分析 (Main)
针对单次实验的完整人格、策略、时序分析。

```bash
python main.py \
  --rollout_file   /path/to/results_exp.json \
  --persona_file   /path/to/personas.json \
  --output_name    analysis_results
```

### 6.2 Unique Segment 深度分析 (Unique)
侧重于分析分支后的独立表现（Unique Segments）以及分支点的策略决策明细。

```bash
python unique_main.py \
  --rollout_file   /path/to/results_exp.json \
  --mode           dissonance \
  --output_name    analysis_unique
```

### 6.3 多模式深度对比 (Deep)
针对多个实验结果（如不同 Intervention 策略）进行横向对比分析。

```bash
python deep_main.py \
  --base           /path/to/results_dir/ \
  --files          results_dissonance.json results_random.json results_none.json \
  --output         deep_comparison_v1
```

---

## 七、新增分析内容说明

### 7.1 Unique Segment 分析器
- **目的**：根轨迹从 Round 1 开始统计，分支轨迹仅从其分叉轮次（BranchTurn）开始统计，从而评估“干预后”的真实动态变化，不被父轨迹历史稀释。
- **输出图表**：`unique_metrics_dashboard.png` (3x2 面板，含 Hs/Ha 及其增量和 Z-score)。
- **报告**：`unique_trends.json` (均值/标准差), `branch_decisions.json` (含策略 reasoning 和分叉点熵值)。

### 7.2 Multi-Mode Deep 对比
- **Intervention Timing**：分析在第几轮进行分支干预的成功率最高。
- **Mode Comparison**：直观对比不同模式下的改善程度（Root vs Oracle vs Rescue Rate）。
- **User Consistency**：分析不同用户在不同模式下的响应差异。
