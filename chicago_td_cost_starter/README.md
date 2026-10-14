# Chicago 道路+交通双模态：离线时变路段成本第一版

> 前置审查：`safe_mapping_summary.json` 显示 71 条安全候选监测路段、507 条唯一 OSM 有向边、0 条冲突。**这里的 safe/strict 指几何筛选和冲突排查，不代表人工核验真值。**

## 输入数据（使用你之前生成的真实字段）

- `datasets/raw/chicago/chicago_loop_drive.graphml`：原先下载、用于匹配的同一份有向OSM路网
- `datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv`
- `datasets/processed/chicago/traffic_long.csv`
- `datasets/processed/chicago/time_index.csv`

依赖：Python 3.10+，`numpy pandas networkx`，**服务器不需要互联网**。若OSM图是由不同版本程序重新导出、edge key发生变化，本脚本会直接报错停止，不会悄悄将速度映射到其他道路。

## 1. 构建时变边权

将此工具包解压至 `~/code/WZL/road_vrp/chicago_td_cost_starter`，在项目根目录执行：

```bash
python chicago_td_cost_starter/build_dynamic_edge_costs.py \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --mapping datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv \
  --traffic-long datasets/processed/chicago/traffic_long.csv \
  --time-index datasets/processed/chicago/time_index.csv \
  --output-dir datasets/processed/chicago/dynamic_costs
```

生成文件：

- `edge_index.csv`：每个OSM有向边对应的稳定`edge_idx`, `osm_u,v,key`、长度、**静态估计**速度/耗时、监测路段ID、映射路径占比
- `time_index.csv`：原始1008个本地时间片顺序
- `edge_costs.npz`：`travel_sec_raw[T,E]`、`travel_sec_fifo[T,E]`、`chicago_observed_mask[T,E]`、`is_mapped_edge[E]`、`slot_local_naive_ns[T]`、`slot_seconds`。**同一份edge_index.csv才能正确解读tensor列顺序。**
- `observed_edge_coverage_by_slot.csv` / `observed_edge_coverage_by_day.csv`：真正落到OSM边的时间有效覆盖率
- `dynamic_cost_summary.json`：质量统计、FIFO修正数量与范围

真实旅行时间转换：

`OSM_edge_TT(t) = Chicago_monitor_segment_TT(t) * (OSM_edge_length / OSM_matched_path_length)`

一段的OSM所有匹配路径时间和与监测段估计时间相同。**注意：这只是在验证映射后可用的初始假设，并非每条OSM道路的独立实测时间。**

- `chicago_observed_mask=True`：有该时间片的监测走廊速度推算而来的边时间；只在这些单元上定义真实监督。
- `is_mapped_edge=True`，`chicago_observed_mask=False`：这条边有对应的监测走廊，但当时没有速度观测；使用**估计静态回退**。
- `is_mapped_edge=False`：没有任何匹配交通监测记录的OSM边，使用**估计静态回退**。
- `travel_sec_raw`：原始样本上的上述混合时间，**不保证FIFO**。
- `travel_sec_fifo`：把 `arrival(t_i)=t_i+cost(t_i)` 做累计最大单调包络；需在`tdsp_probe.py`中按时间对**arrival函数进行线性插值**，保证采样点与插值时刻满足FIFO。FIFO修正时间不是原始观测值，不得用于交通预测监督。

静态回退速度优先采用GraphML上已有`speed_kph`，否则取`maxspeed`（无单位则按芝加哥mph惯例），否则按道路类别估计。以上均属于**路由/插值假设**，不可当作真实交通速度。

## 2. 验证时间依赖最短路径

在相同的离线环境运行：

```bash
python chicago_td_cost_starter/tdsp_probe.py \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --departure '2025-06-05 08:00:00' \
  --num-pairs 30 \
  --output-csv datasets/processed/chicago/dynamic_costs/tdsp_od_probe.csv
```

会从OSM最大的强连通分量随机抽30对OD，分别：

1. 用静态估计成本求最短路，再在**同一个历史真实交通场景+回退成本**下重放评价。
2. 使用**已知整段历史交通**得到FIFO时间依赖最短路（oracle/hindsight）。

输出比对结果包括时变最短路可改善多少秒、其路径包含多少个有观测的边。若多数OD完全不触及监测边，说明目前71段、507边的交通覆盖无法有效支撑VRP交通决策，应改选仓库和客户区域、扩大时间数据/匹配范围，不能继续盲目训练GNN。

**此脚本尚不是多车CVRP求解器。** TD-CVRP还需固定客户位置/需求/车辆容量以及求解器、路径缓存、服务时间和有可行性保证的时间窗。该probe只用于验证路由器和估计决策价值。

## 3. 关键实验方法论

- 所有输入时间仍采用 Chicago 原始`time`预处理后**未验证时区的本地墙上时间**。在高峰时段实验前核实官方时区，并保留夏令时处理记录。
- 离线历史真实速度可以用于**回测标签和事后oracle基准**，不能当作未来时刻的可用实时预测特征。真正的Road+Traffic学习输入只允许使用决策时刻已知的历史窗口和静态道路。
- 如果 `dynamic_cost_summary.json` 中`fraction_observed_within_entire_graph_grid`很低，必须报告，因为其它路段由静态估计填补，不能宣称完整真实动态路网。
- 抽查建议：地图核验不少于20条严格候选监测路段，重点看方向、几何和长度比；建议更长期将所有71条都审查一遍。
- 划分训练验证测试时按**连续日期块**且历史输入窗口不得越过划分边界。7天仅作先导试验，不适合声称稳定泛化结果。

## 4. 验证脚本

`test_synthetic.py` 使用合成GraphML和交通速度测试无冲突路段映射、观测Mask、缺失回退、旅行时间守恒、FIFO以及TDSP运行。测试不代表已在你的真实CSV/GraphML上成功运行；正式运行时请检查输出summary并反馈。
