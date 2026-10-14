# Chicago真实路网 TD-CVRP30：多求解器比较与消融工具包

本包与现有项目的数据结构兼容，**不会覆盖之前的 `tdcvrp30.py`、`tdcvrp/` 或历史OD回归文件**。仅使用 `numpy pandas networkx`，全部操作可在不能访问外网的Ubuntu服务器上完成。原Chicago交通数据/道路GraphML不包含在压缩包内。

## 0. 实现范围

- **基线求解器（无需模型训练）**：`greedy`（最远客户种子+最便宜插入）、`sa`（模拟退火随机邻域）、`vnd`（多邻域首次改进局部搜索）、`alns`（自适应随机/最坏/相关性破坏+贪心/Regret-2修复+局部改进）。
- **ALNS消融版本**：`alns_no_related`、`alns_no_worst`、`alns_greedy_repair`（关闭regret2）、`alns_uniform`（关闭自适应选择）、`alns_no_local`（关闭局部改进）。
- **两种规划成本模式**：`static`（仅OSM静态估计）与`oracle`（历史未来交通已知，**仅用于事后参考/诊断**）。两者最终都在**同一真实路网的历史+静态回退边成本模型**上，按客户顺序逐OD执行时间依赖Dijkstra进行回测。
- **车辆约束**：`exactly`（必须使用全部5辆）与 `at-most`（最多5辆，允许空路线）。不可混用约束比较。
- **目标**：最小化车队**总行驶时间**（非总车时、非最大行程/公平性）；服务时间只用于推进下一客户的到达时间。当前不考虑客户时间窗。
- **搜索预算**：每种算法相同 `--wall-sec` 和 `--max-iterations` 上限；Greedy一次构造后即返回、VND局部最优即停止；记录实际`search_wall_sec`、`construct_wall_sec`、`score_calls`、`iterations`。**不能把ALNS一次迭代视为与SA一次迭代等价工作量**。
- **结果**：`metrics.csv`、`paired_gains.csv`、`solutions.json`、`benchmark_summary.json`，再用`summary`脚本输出算法对比统计，`teacher_pool`脚本产生近似教师和跨交通情景后悔值。

现有Chicago道路图只有一部分监测动态速度，其余边/时段使用静态估计，**不是完整真实动态路网**；“oracle”也并非可部署的预报系统或数学全局最优。

## 1. 上传和解压

```bash
cd ~/code/WZL/road_vrp
unzip chicago_tdcvrp_solver_suite.zip
```

确认已经存在此前处理好的：

```text
datasets/processed/chicago/dynamic_costs/
  edge_index.csv
  edge_costs.npz
  time_index.csv

datasets/processed/chicago/tdcvrp30/tasks/
  task_0000.json
  task_0001.json
  ...
```

Python环境使用你已有的 `WZLroadvrp`；确认有 `numpy pandas networkx`。在服务器断网时请使用本地预装依赖或上传离线wheel，不能假设pip能够在线安装。

## 2. 先跑合成测试（无需Chicago文件）

```bash
cd chicago_tdcvrp_solver_suite
python -m tests.test_synthetic
cd ..
```

这项测试在自动构造的36个节点有向网格上检验9种方法×两种规划模式、容量约束、恰好K与最多K两种设置，以及精确道路回测。**它不是Chicago真实性测试**。

## 3. 第一轮：四种主要求解器（1任务×2天×5种子）

在项目根目录：

```bash
python chicago_tdcvrp_solver_suite/benchmark_suite.py \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --output-dir runs/chicago_solver_suite/base \
  --od-cache-dir runs/chicago_solver_suite/od_cache \
  --methods greedy,sa,vnd,alns \
  --planners static,oracle \
  --seeds 42,43,44,45,46 \
  --max-tasks 1 --max-scenarios 2 \
  --vehicle-rule exactly \
  --od-step-min 20 --od-hours 4 \
  --max-iterations 5000 --wall-sec 5
```

输出每条记录：`solver,planner,seed,task_id,scenario_id,ex_post_road_travel_sec,observed_edge_share,vehicle_loads,construct_wall_sec,search_wall_sec,score_calls,...`。

**如预算实验以纯时间为主**：将`--max-iterations`设得很大（如1000000），以`--wall-sec`控制实际运行时间；`greedy`和已收敛的`vnd`不会人为空转，必须报告实际用时。构造时间与OD预计算时间始终分别记录。

## 4. 第二轮：ALNS算子消融（保持同样实例、种子、预算）

```bash
python chicago_tdcvrp_solver_suite/benchmark_suite.py \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --output-dir runs/chicago_solver_suite/ablations \
  --od-cache-dir runs/chicago_solver_suite/od_cache \
  --methods alns,alns_no_related,alns_no_worst,alns_greedy_repair,alns_uniform,alns_no_local \
  --planners oracle \
  --seeds 42,43,44,45,46 \
  --max-tasks 1 --max-scenarios 2 \
  --vehicle-rule exactly \
  --od-step-min 20 --od-hours 4 \
  --max-iterations 5000 --wall-sec 5
```

这组只回答**启发式搜索模块**的贡献，不要与以后`Road GNN`/`Traffic Encoder`/`InfoNCE`的贡献混淆。后者必须**固定求解器、时间预算、输入信息可用性和任务划分**进行单独消融。

## 5. 分析与汇总

```bash
python chicago_tdcvrp_solver_suite/summarize_results.py \
  --metrics runs/chicago_solver_suite/base/metrics.csv \
  --output-dir runs/chicago_solver_suite/base/summary
```

得到：
- `solver_summary.csv`：平均道路级行驶时间、离线候选池差距（**非理论optimality gap**）、平均求解/OD预计算用时、目标评估调用数、观测道路比例。
- `paired_gain_by_seed.csv`：每种子和场景`static`与`oracle`的**配对**节省率，负值不能删。
- `per_instance_stability.csv`：同一个任务+交通情景内先对多个种子求均值/标准差/胜率。
- `paired_summary_by_solver.csv`：再跨独立任务+情景汇总（同一任务跨天不独立，正式论文仍需按任务层面做区组统计）。

为获得稳健结论：固定任务和时间条件，至少5个种子，报告**均值、标准差、负收益比例**，而不是只展示最佳解。建议先10任务×6情景，稳定后扩展40×6。由于大量精确TDSP回测较昂贵，先小规模试跑并监控时间。

## 6. 教师候选池和后悔值（所有求解器共同的历史参考）

```bash
python chicago_tdcvrp_solver_suite/teacher_pool.py \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --solutions runs/chicago_solver_suite/base/solutions.json \
  --output-dir runs/chicago_solver_suite/teacher_pool
```

得到各情景池内最优近似参考解`teachers.json`、`candidate_costs.csv`和`regret_pairs.csv`。会把来自**所有求解器、所有种子、同一任务不同交通日期**的客户分组+访问顺序统一交叉回测，避免把不同启发式的随机噪声简单误认为决策后悔值。使用此前固定同钟点的场景。

注意：池内最佳不一定是真实最优；跨情景互评需要历史事后信息，**只能生成离线训练标签和事后分析，不能当作在线模型输入**。正式训练集/测试集需严格按任务及日期分开，不能在测试集内用标签挑选最终模型。

## 7. 在现有地图上对比算法路线

工具包另有导出脚本，将多求解器的`solutions.json`转成你之前`visualize_tdcvrp_routes.py`兼容的格式：

```bash
python chicago_tdcvrp_solver_suite/export_map_solutions.py \
  --solutions runs/chicago_solver_suite/base/solutions.json \
  --solver alns --seed 42 \
  --output runs/chicago_solver_suite/map_alns_seed42.json
```

然后沿用你已有的离线HTML地图生成器，仅替换`--solutions`为这个文件即可：

```bash
python chicago_route_map_offline/visualize_tdcvrp_routes.py \
  --solutions runs/chicago_solver_suite/map_alns_seed42.json \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --output runs/chicago_solver_suite/map_alns_seed42.html
```

**需要使用`--vehicle-rule exactly`生成地图方案**，以兼容旧地图脚本的强制非空车辆检查。另为`sa`、`vnd`导出不同文件，即可放在同一浏览器中观察路线改善。此前地图脚本为独立工具，此ZIP不包含地图源码。

## 8. 后续多模态与对比学习消融设计

| 消融主轴 | 对比组 | 固定不变的条件 |
|---|---|---|
| 求解器算法 | Greedy / SA / VND / ALNS | 相同道路、交通、任务、预算 |
| ALNS搜索模块 | 去相关性/最坏移除、只Greedy插入、统一算子权重、无局部改进 | 交通信息与其余算子不变 |
| 规划信息 | static / hindsight oracle / **未来真正预测的traffic** | 相同ALNS和精确事后评价器 |
| 神经表示 | Road-only / Traffic-only / 双模态融合 | 相同教师、训练数据切分、Decoder与预算 |
| Contrastive | 无CL / InfoNCE / 决策后悔值监督CL | 相同双模态骨干与求解器 |
| 泛化 | 同城新日期 / 新客户分布 / 新规模 / 新区域 | 固定数据可用性，报告缺失掩码覆盖率 |

关于模型公平性：Road-only与Traffic-only输入特征不等价，可以用消融判定信息增益，但其计算开销和参数量也应报告；未来交通信息必须由历史窗口预测，**绝不能直接把事后oracle速度作为测试输入**。

## 9. 可能的异常

- `Route time exceeds sampled OD horizon`：增加`--od-hours`（不能超过可用历史交通序列边界），不应直接截断成本。
- `Expected 5 routes`、`Capacity violation`：检查新旧任务是否相同，不能随意覆盖`task_0000.json`。
- `No matching task_*.json`：将`--task-dir`指向此前生成任务目录。
- `Unreachable OD`: 检查仓库/客户是否都在OSM最大有向强连通分量内。
- 结果`oracle`仍劣于`static`：启发式搜索近似和OD插值误差是可能原因，**不等于底层TDSP错误**；要使用精确道路回测和多种子池分析。
- OSM历史速度仅部分覆盖：观察`observed_edge_share`并报告未观测道路的静态回退假设。
- 大量组合运行较慢：建议先`--max-tasks 1 --max-scenarios 1 --seeds 42 --methods greedy,sa`排查，再扩容。

## 测试范围

已经使用**合成6×6网格**跑通所有9种方法×两种规划模式×两种历史情景，检查容量和严格客户覆盖；并对一个任务的多算法候选执行完整路网回测和跨情景教师评估。此测试**不能替代你的真实Chicago GraphML和边权文件上的验收**。
