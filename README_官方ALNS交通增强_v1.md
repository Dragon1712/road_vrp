# Chicago TD-CVRP30：官方ALNS搜索增强 + 交通感知算子（v1）

**官方开源框架**：N-Wouda/ALNS，MIT License，https://github.com/N-Wouda/ALNS ，目标接口 `alns==7.0.0`。本补丁只提供自研Chicago特定算子和benchmark适配器，**不分发、修改或复制官方库的源码**。如计划再分发官方库，遵循其 MIT 原版权声明。

**与现有项目兼容**：只新增文件到 `chicago_tdcvrp_solver_suite_v2/`，旧版`upstream_alns_adapter.py`、`solvers.py`、`benchmark_suite.py`、已有`runs/...`不被覆盖或修改。要求项目根目录已有v2代码、官方ALNS桥接器、`datasets/processed/chicago/dynamic_costs/{edge_costs.npz,edge_index.csv,time_index.csv}`及10任务数据。未来交通Oracle仍然只是事后上界参考，非可部署预测。

## 1. 新增方法（实验消融设计）

| method | 搜索模块 | 交通特征 | 研究用途 |
|---|---|---|---|
| `upstream_alns` | 旧版官方ALNS适配器 | 无额外交通特征 | 已有基线，旧代码提供 |
| `upstream_plus` | 连续片段移除 + Regret-3 + 随机局部搜索，6种destroy、3种repair | **仅静态道路特征** | 搜索变强带来的增益 |
| `upstream_traffic` | 与`upstream_plus`相同搜索结构 | 交通相关性移除 + 交通瓶颈移除 | 交通信息额外增益 |
| `upstream_traffic_no_t_related` | 同上 | Traffic-Related替换为Static-Related | 去交通相关性消融 |
| `upstream_traffic_no_t_worst` | 同上 | Traffic-Worst替换为Static-Worst | 去交通瓶颈消融 |
| `upstream_traffic_no_regret3` | 修复从3种降到2种 | 有交通特征 | 去Regret-3消融 |
| `upstream_traffic_no_local` | 不进行随机化局部搜索 | 有交通特征 | 去局部改进消融 |
| `upstream_traffic_uniform` | 随机均匀选算子而不自适应 | 有交通特征 | 去自适应选择消融 |

**公平性**：`upstream_plus`与`upstream_traffic`有**相同数量与类型槽位**的移除/修复算子，只有两个静态信号槽位被替换为相应交通信号。均调用同一v2 `Objective`、同一OD缓存、`road.exact`以及相同预算限制；故二者的差异最接近“额外交通信号的贡献”，尽管两者的随机搜索轨迹与算子权重仍可能不同。

## 2. 两个交通感知算子

### Traffic-Related (Traffic-aware Shaw removal)

使用矩阵 `ODCost.dynamic[k,i,j]` 在历史交通情景出发时刻的**OD时变最短路成本**，与静态OD作为对照：

- `d_dyn(i,j) = (TDSP(t0,i,j) + TDSP(t0,j,i))/2`，仅用于计算移除相关性，**不用于替代有向OD真实评价**。
- 为每个客户选择静态邻近的最多6个目的地，计算出发时刻OD交通延误比例 `mean((dyn[0,i,j]-static[i,j])/static[i,j])` 与其沿整个采样时间网格的时间标准差。
- 结合客户需求差异与原路线所属车辆相同的奖励，得到Shaw相关性分数，优先移除关系较密切的客户集合，以探索跨车调度。

这是**OD级交通特征**，不是“每条路段均有实测”的断言。依然受当前实测边时格点约6.4%的支持率影响。

### Traffic-Worst (time-conditioned hotspot removal)

首先按当前完整时变路线的**边际移除节约**评价客户（调用`obj.route_score`），再对其在计划访问时间实际到达的前后OD段计算正向延误：

`hotspot(c) = max(0, t_dyn(prev,c,t)-t_static(prev,c)) + max(0, t_dyn(c,next,t_arrival+service)-t_static(c,next))`

并使用 `benefit(c) + 0.45 * hotspot(c)` 排序，偏向优先移除穿越相对拥堵OD的客户；从最重要的前三分之一随机抽样，保持多样性。

`static_worst`只使用纯静态OD局部弧差 `static(prev,c)+static(c,next)-static(prev,next)`，不叠加`hotspot`。另一个普通`worst`算子才使用当前完整的目标函数边际移除节约。注意：在Oracle规划模式下，**两组方案的主优化目标都已经使用动态OD**；实验比较的是“显式拥堵引导算子”的额外贡献，而不是“有没有动态交通信息”的全部贡献。

## 3. 搜索增强

- `segment`：移除车辆访问序列上连续的一段客户，为原先逐点移除补充成块调整；Exactly-K下保留每辆车至少1个客户。
- `regret3`：逐步计算每个未分配客户的容量可行插入位置，再按最佳、第2、第3插入成本差的和排序并执行最佳插入。所有插入成本调用完整路线`route_score`以保留时间依赖，不能简单使用静态弧差。
- `local_search`：每4次修复执行最多36次随机邻域尝试（Relocate、Swap、Reverse、2-opt*），找到首次严格改善即返回。这是有限预算局部搜索，非VND全邻域局部最优保证。
- `CombinedStop`：保持官方ALNS框架的停止接口与5秒/5000次上限，并单独记录评分调用次数和修复失败次数。
- 任一修复无法组成可行解，则**返回原可行父状态**。所有方法必须验证30客户恰好访问一次、载重不超过45、Exactly-5设置下5辆车均非空。

## 4. 服务器安装与合成验收

压缩包解压到项目根目录：

```bash
cd ~/code/WZL/road_vrp
unzip chicago_traffic_alns_v1.zip
conda activate WZLroadvrp
python -c 'from alns import ALNS; from alns.select import RouletteWheel; from alns.accept import SimulatedAnnealing; print("Official ALNS is ready")'
cd chicago_tdcvrp_solver_suite_v2
python -m tests.test_traffic_alns
cd ..
```

如果未安装官方ALNS：`python -m pip install 'alns==7.0.0'`；离线服务器请在相同系统/兼容Python环境先下载wheel及全部依赖，用`pip install --no-index --find-links /path/to/wheels 'alns==7.0.0'`安装。**不要在服务器离线环境直接执行`pip install`并假设能联网。**

## 5. 先做Smoke

```bash
bash chicago_tdcvrp_solver_suite_v2/run_traffic_alns.sh smoke
```

会测试1任务×1日期×1种子×8配置。结果在 `runs/chicago_solver_suite_v2/traffic_alns/smoke/`。第一次需运行，文件已存在后须设置`RESUME=1`，否则故意拒绝覆盖。

## 6. 搜索增强与交通增强的两级比较

```bash
bash chicago_tdcvrp_solver_suite_v2/run_traffic_alns.sh compare
```

默认是10任务×6日期×3种子×3方法=540条记录，分别研究：

- `upstream_alns` vs `upstream_plus`：增强搜索算子是否更好？
- `upstream_plus` vs `upstream_traffic`：引入明确的交通信号是否提供额外增益？

输出：

- `traffic_alns/compare/metrics.csv`（道路级TDSP精确回测）
- `traffic_alns/compare/solutions.json`（客户序列和详细算子诊断）
- `traffic_alns/compare/vs_search_only/solver_comparison.csv`（与upstream_plus配对）
- `traffic_alns/compare/vs_traffic_full/solver_comparison.csv`（与upstream_traffic配对）

**方向**：`compare_official_alns.py` 的`gain_sec = J_reference - J_method`。若某行method为`upstream_traffic`、reference为`upstream_plus`，正值表明交通算子更好。请优先分析任务级结果，而不是只报告180次随机运行的方差。

## 7. 交通算子与搜索模块消融

```bash
bash chicago_tdcvrp_solver_suite_v2/run_traffic_alns.sh ablations
```

默认10任务×6情景×3随机种子×7方法=1260条；包括Full、去交通Related、去交通Worst、去Regret-3、去局部、去自适应权重以及静态增强搜索配置。另有针对前述官方ALNS基线的`compare`阶段。默认每方法搜索预算5秒，**合计大量运行约需小时级**（依服务器性能）。

临时更改运行预算/规模，例如只测试2任务、2日期、种子42且1秒：

```bash
TASKS=2 SCENARIOS=2 SEEDS=42 BUDGET=1000 WALL=1 \
  bash chicago_tdcvrp_solver_suite_v2/run_traffic_alns.sh compare
```

**切勿**改变参数后继续沿用旧输出目录。要完全新实验，请移动旧目录到归档路径，或修改脚本里的OUT为新名字。相同配置断点恢复可使用：

```bash
RESUME=1 bash chicago_tdcvrp_solver_suite_v2/run_traffic_alns.sh compare
```

## 8. 算子诊断和路线可视化

```bash
python chicago_tdcvrp_solver_suite_v2/summarize_traffic_operators.py \
  --solutions runs/chicago_solver_suite_v2/traffic_alns/compare/solutions.json \
  --output-dir runs/chicago_solver_suite_v2/traffic_alns/compare/operator_analysis
```

输出算子使用次数、失败修复次数、动态特征均值、局部改善数量。观察`dynamic_signal_enabled`是否True、交通算子是否真的被调用以及traffic gain是否随客户交通变化而增加。

使用此前交付的`chicago_route_map_v3/visualize_solver_routes.py`直接读取上述`solutions.json`，令`--solvers upstream_plus,upstream_traffic`，其它`--task-dir/--cost-dir/--graphml`参数维持原值；无需数据转换。建议先画`task_0000`在6月5日的路线，再选一个动态交通增益最大的任务，避免只展示有利案例。

## 9. 研究可信度边界

1. **无真实Chicago性能结论**：本补丁在当前环境通过合成有向路网验证，但本环境未安装官方`alns`PyPI包，也未在用户真实数据上跑过；需服务器实际运行输出证明性能。
2. **严格杜绝测试信息泄漏**：`upstream_traffic`在`oracle`模式使用未来已知历史成本，只是**离线hindsight teacher**；`static`模式自动降级为只用静态交通特征。以后训练预测模型时，需要用**出发前可观测信息**预测同形状OD张量，然后才运行相同traffic算子，绝不可在测试时传真实未来交通。
3. **时变OD插值**：搜索近似用20分钟采样OD，而最终准确回测逐OD使用时变OSM最短路。仅对当前观测/静态回退成本模型精确，不能视为真实全城交通真值。
4. **固定时间预算 ≠ 固定搜索工作量**。不同修复算子消耗不同，建议同时分析单位时间最好成本和score_calls，并增加1秒、3秒、5秒、10秒预算曲线。
5. **方差与样本量**：10个任务×6日期共180次运行，只有10个独立订单任务，同一城市7天数据；跨情景非独立，不能声称跨城市或长期泛化。
6. **负收益必须保留**：新的算子可能在某些任务上更差，不能筛掉或声称必然提高。建议先检查增益分布、任务级Bootstrap区间、交通增益是否随实测覆盖比例上升。
7. **没有时间窗**：当前是TD-CVRP30，不是TD-CVRPTW；当前结果不代表有严格配送时间窗、车辆固定成本或实时需求的场景。

## 10. 新增文件清单

```text
chicago_tdcvrp_solver_suite_v2/
  traffic_alns_adapter.py         新的搜索与交通感知算子
  benchmark_traffic_alns.py       给已有benchmark添加方法入口
  run_traffic_alns.sh             smoke/compare/ablations实验脚本
  summarize_traffic_operators.py  统计每种算子真实使用情况
  tests/test_traffic_alns.py      容量、客户覆盖、无泄漏、合成成本、完整IO测试
```

**下一步**：先运行Smoke并上传`metrics.csv`及`solutions.json`中`solver_diagnostics`统计。然后按10任务全量跑`compare`和`ablations`。如计划写论文，必须增加真实官方库验收、交通数据遮罩鲁棒性和跨任务统计显著性审查。
