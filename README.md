# Chicago真实路网 + 历史交通：TD-CVRP30离线先导实验包

这是**可以在无法联网的Ubuntu服务器运行的实现**，与之前的 `chicago_td_cost_starter` 输出格式直接对接，独立脚本入口为 `tdcvrp30.py`。

## 已知输入与界限

你目前的 `dynamic_cost_summary.json` 显示 **2265节点、5503条有向边、71条安全候选监测走廊、507条有映射边、1008个10分钟时间片**。映射边约69.47%时间有有效观测，但整个图在时间×边格点中只有**6.40%具有Chicago动态数据支持**。未观测边/时段采用OSM静态参考成本，不得描述为实测动态交通。

**第一版只研究 TD-CVRP30（容量、单仓、5车且每辆车出车、无时间窗）。** 先排查地图方向等匹配质量，后面再扩展CVRPTW、鲁棒性与真正的在线交通预测。这里的道路+交通是供后续模型学习的两模态，但目前**不包含任何已训练的多模态预测模型**。

依赖：`Python 3.10+`，`numpy`、`pandas`、`networkx`。依赖需在服务器已有的虚拟环境中安装，脚本本身完全离线。没有使用OR-Tools，不声称最优性。

## 0. 解压和检查路径

从项目根目录：

```bash
cd ~/code/WZL/road_vrp
unzip chicago_tdcvrp30_offline.zip
python tdcvrp30.py -h
```

需要已经存在：

```text
datasets/processed/chicago/dynamic_costs/
├── edge_index.csv
├── time_index.csv
└── edge_costs.npz
```

确保与此前跑`tdsp_probe.py`时使用的是**同一个边编号、同一份成本张量**。本ZIP内附之前上传的`fixtures/tdsp_morning.csv`与`fixtures/tdsp_evening.csv`，各100条OD，是回归测试的参考，不是VRP训练集。

## 1. 运行原有100组OD的回归测试【先做】

```bash
python tdcvrp30.py regress \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --morning fixtures/tdsp_morning.csv \
  --evening fixtures/tdsp_evening.csv \
  --output-dir runs/chicago_tdcvrp30/regression
```

会重新计算**200次OD比较**（早、晚各100）：`static_planned_sec`、`static_route_realized_sec`、`oracle_td_route_sec`、`oracle_improvement_sec`。默认误差阈值0.01秒（原CSV保留3位小数）。`oracle_td_edges`、观测边数额外报告一致性，但最短路平局可能造成路径不同，故作为诊断而不是判定失败条件。

输出：
- `runs/chicago_tdcvrp30/regression/tdsp_regression_report.json`
- `runs/chicago_tdcvrp30/regression/tdsp_regression_details.csv`

**必须先满足 `time_cost_all_pass=true` 才进入VRP求解**。如果未通过，很可能不是优化算法的问题，而是底层图、边索引、时间解析或动态成本的版本不一致。

## 2. 生成30客户×5车任务（同任务多个交通情景）

```bash
python tdcvrp30.py generate \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --output-dir datasets/processed/chicago/tdcvrp30/tasks \
  --num-tasks 10 --customers 30 --vehicles 5 \
  --capacity 45 --service-seconds 120 \
  --strategy mixed --radius-m 600 --seed 2026
```

生成： `task_0000.json` ... `task_0009.json`、`manifest.json`。每个任务固定30个客户OSM节点、同一仓库、固定客户容量需求与5车辆容量可行初始解；每个任务有**6个日期情景，均在08:00出发**（2025年6月3～8日，优先测试6月5日及6日），有助于同钟点的交通影响比较。全部合成客户都从**最大有向强连通分量**选取。`mixed`表示约70%的地点靠近动态监测道路，其余在全区随机采样；用来增加决策敏感性，不等于真实物流订单分布。

如需观察不同阶段订单与交通覆盖影响，可另外用 `--strategy corridor` 或 `--strategy uniform` 生成一组独立数据。不要只展示针对监测走廊刻意采样的最好结果。

## 3. 首次小规模求解（1任务×2日期）

```bash
python tdcvrp30.py solve \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --cache-dir runs/chicago_tdcvrp30/od_cache \
  --output-dir runs/chicago_tdcvrp30/pilot \
  --max-tasks 1 --max-scenarios 2 \
  --od-hours 4 --od-step-minutes 20 \
  --iterations 1500 --restarts 2 --seed 42
```

- **静态方案**：OSM静态OD估计成本下选择5车的客户分配与访问顺序。
- **Oracle方案**：历史交通全部已知条件下，使用经20分钟OD采样网格近似的时变OD评价选择5车的客户分配与访问顺序。
- **共同回测**：两者的配送顺序都在**同一个历史情景**中，使用原始路网`edge_costs.npz`上**逐段TDSP准确重路由**得到`ex_post_road_travel_sec`。这是对**客户访问顺序/车辆分配**的公平比较，**不是**“静态道路路径固定不变 vs 动态道路可重路由”的比较。

`--iterations`为候选局部搜索迭代次数，两方案使用相同预算；求解器是多启动模拟退火式局部搜索（relocate/swap/2-opt），不是已证最优算法。所有车辆必须承担任务，没有时间窗，出车时刻相同。搜索阶段通过采样OD到达函数插值加速，因此搜索时的`planned_travel_sec`与最后真实路网重算的`ex_post_road_travel_sec`通常**不会完全相同**。必须以**真实路网重算**指标评估。

输出：
- `runs/chicago_tdcvrp30/pilot/metrics.csv`：每任务、每情景两种方案的同情景回测结果。
- `runs/chicago_tdcvrp30/pilot/solutions.json`：具体5辆车的客户序列、容量、每个客户的到达时间与跨边的观测统计。
- `runs/chicago_tdcvrp30/pilot/run_summary.json`：参数设置与方法限制。
- `runs/chicago_tdcvrp30/od_cache/*.npz`：每任务每日期的客户OD旅行时间网格缓存，可复用。

**若出现 dynamic Oracle比静态基线更差：不是Bug的充分证据。** 因为优化器为近似算法，OD网格也有插值误差；需要增大迭代预算、细化OD时间采样、增加随机种子和比对离线真值。**不可只保留正改善的任务**。

## 3.1 检查OD采样近似误差【建议首次求解后做】

```bash
python tdcvrp30.py od-audit \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --cache-dir runs/chicago_tdcvrp30/od_cache \
  --output-dir runs/chicago_tdcvrp30/od_audit \
  --samples 100 --od-hours 4 --od-step-minutes 20
```

结果`od_approx_summary.json`提供**非整采样网格时刻的OD插值误差**。若平均/尾部误差较大，增加OD密度，比如将`--od-step-minutes 20`改为`10`。不要只核对“20分钟采样节点”处，因为那里误差天然接近0。需要注意缓存带有采样步长标识，程序会重新计算不同粒度的OD表。

## 4. 扩展到10任务×6情景

```bash
python tdcvrp30.py solve \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --cache-dir runs/chicago_tdcvrp30/od_cache \
  --output-dir runs/chicago_tdcvrp30/pilot_full \
  --max-tasks 10 --max-scenarios 6 \
  --od-hours 4 --od-step-minutes 20 \
  --iterations 1500 --restarts 2 --seed 42
```

先用`--max-tasks 1`或`--max-scenarios 2`试跑。全部任务会预计算至少**60组×31起点×13时间点**的时间依赖最短路，服务器端可能需要相当长的计算时间。缓存完成后第二次运行会明显加快。任务只接受在输入交通数据范围内的时刻。

## 5. 生成初步跨情景后悔值（不是决策CL论文结论）

```bash
python tdcvrp30.py regret \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --solutions runs/chicago_tdcvrp30/pilot_full/solutions.json \
  --output-dir runs/chicago_tdcvrp30/regret
```

同任务、同出发钟点、不同交通日期，分别计算`solution_i`在`scenario_j`下的真实路网回测成本（仍是**历史回溯成本+静态回退边**）。输出 `cross_costs.csv`、`regret_pairs.csv`、`regret_summary.json`，计算正向、逆向和对称归一化的近似后悔值。求解参考不是全局最优，可能出现负的原始“后悔值”，代码会显式保留raw差值，并在相似度计算时截断为0；后续训练需要多随机种子/更可靠参考求解器。

## 6. 离线单元测试

```bash
python -m tests.test_synthetic
```

会在临时目录生成**合成**8×8双向格点图与自造交通张量，验证FIFO、最大强连通分量采样、车辆容量、所有客户覆盖、两种规划模式与最终道路成本重算。**合成测试通过不代表真实Chicago数据上已验证成功。**

## 7. 科研诚信与消融协议

1. `Chicago_observed_mask`为True才有历史监测速度标签；其余OD路线包含静态参考值，不能称100%真实交通。
2. **不能把Oracle交通张量当作测试时可用的预测信息**。本版本只创建优化问题与近似高质量参考解。后续Road GNN+Traffic Encoder需以出发前的历史窗口产生未来成本预测，再用相同road-leg evaluator比较。
3. 先固定交通日期划分、地点分布、求解预算和随机种子，再比较Road-only、Traffic-only、双模态、InfoNCE和DecisionCL。数据目前只有7天：允许Pilot，**不足正式跨日期泛化结论**。
4. 对使用映射几何检验的71条监测走廊进行地图核验抽查（最好全部核查）；不能仅用几何`strict`标签称“真实正确”。
5. 记录输入时区：当前`time_local`没有最终验明TZ；尤其跨DST季节重新采集时必须显式转时区。
6. 既有100 OD回归只证明**道路时间成本/TDSP计算逻辑**未被后续VRP模块改变；不会证明VRP启发式求解质量。

## 8. 一键先导运行

如希望自动依次完成200个OD回归测试、10个任务生成、1任务×2情景求解、OD近似误差审查，在项目根目录执行：

```bash
bash run_stage1.sh
```

脚本采用`set -euo pipefail`：**若200个OD参考时间回归失败，后续VRP求解不会继续运行**。`fixtures/dynamic_cost_summary_for_reference.json`为此次用户实际提供的输入质量参考，代码不将其当作机器学习标签或对运行结果强行赋值。
