# Chicago TD-CVRP30 · 原版ALNS vs 官方ALNS 道路级双图对比（v3）

本工具与 `Dragon1712/road_vrp` 最新v2实验输出兼容，**不需要安装官方ALNS库**就能画图。

数据源：

- `chicago_tdcvrp_solver_suite_v2/upstream_library/compare/solutions.json`：原版ALNS和开源ALNS对照。
- `chicago_tdcvrp_solver_suite_v2/upstream_library/ablations/solutions.json`：开源ALNS及五个消融版本。
- `chicago_tdcvrp_solver_suite_v2/base_10tasks/solutions.json`：Greedy、SA、VND、原版ALNS，static/oracle。
- `datasets/processed/chicago/tdcvrp30/tasks_10/*.json`：固定任务中客户的OSM路网节点、需求和服务时间。
- `datasets/raw/chicago/chicago_loop_drive.graphml`：Chicago Loop道路图及可视化经纬度/边几何。
- `datasets/processed/chicago/dynamic_costs/{edge_costs.npz,edge_index.csv,time_index.csv}`：历史时变边权。

**注意：这是历史路网模拟的时变最短路轨迹，不是GPS车辆实测轨迹。** 不同车辆编号不代表同一辆物理车辆；比较的是五车整体路线与成本。所有求解器最终成本均使用历史时变道路级最短路重回测，因此无论搜索模式static还是oracle，地图路径均使用 `mode=oracle` 路径进行回放。

## 使用方法

将整个 `chicago_route_map_v3` 目录放在 `~/code/WZL/road_vrp` 根目录，并进入该目录：

```bash
cd ~/code/WZL/road_vrp
python -m pip show networkx numpy pandas
```

### A. 官方ALNS vs 原版ALNS：6天×1任务×seed 42

```bash
python chicago_route_map_v3/visualize_solver_routes.py \
  --solutions runs/chicago_solver_suite_v2/upstream_library/compare/solutions.json \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks_10 \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --task-id task_0000 --seed 42 --scenario all \
  --solvers alns,upstream_alns --planner oracle \
  --output runs/chicago_solver_suite_v2/maps/task_0000_seed42_compare.html
```

生成的HTML打开后可以切换6个日期，并左右分别选择原版ALNS与官方ALNS。**默认展示seed=42**，不会隐式取各算法在不同种子中的最优值。若要其他seed，必须明确切换`--seed 43`或`--seed 44`并换新输出文件名。

### B. 只画单个时段，减少运行和HTML体积

```bash
python chicago_route_map_v3/visualize_solver_routes.py \
  --solutions runs/chicago_solver_suite_v2/upstream_library/compare/solutions.json \
  --task-id task_0000 --seed 42 \
  --scenario 2025-06-05_0800 \
  --solvers alns,upstream_alns \
  --output runs/chicago_solver_suite_v2/maps/task_0000_0605_compare.html
```

### C. 官方ALNS五个消融在地图上比较

```bash
python chicago_route_map_v3/visualize_solver_routes.py \
  --solutions runs/chicago_solver_suite_v2/upstream_library/ablations/solutions.json \
  --task-id task_0007 --seed 42 --scenario 2025-06-05_0800 \
  --solvers upstream_alns,upstream_no_related,upstream_no_worst,upstream_greedy_repair,upstream_uniform,upstream_no_local \
  --output runs/chicago_solver_suite_v2/maps/task_0007_ablations.html
```

网页左右下拉框分别选消融版本，比对车辆分组与绕行道路。

### D. 基础求解器/Static vs Oracle

```bash
python chicago_route_map_v3/visualize_solver_routes.py \
  --solutions runs/chicago_solver_suite_v2/base_10tasks/solutions.json \
  --task-id task_0000 --seed 42 --scenario 2025-06-05_0800 \
  --solvers alns,sa,vnd,greedy --planner all \
  --output runs/chicago_solver_suite_v2/maps/task_0000_base.html
```

HTML左/右下拉框的标签含规划信息，选择`alns / static`与`alns / oracle`比较即可。

## 功能

- 真实GraphML路网底图，不使用Mapbox/Folium/在线瓦片/CDN，HTML可脱机打开。
- 左右两幅地图坐标同步缩放、拖拽，并显示同情景的行驶时间绝对和相对差值。
- 五车彩色线路、客户编号、车辆访问顺序和载重、客户到达时间、仓库位置、道路行驶方向箭头。
- 客户区域/全路网一键缩放，逐车辆筛选，点击`保存双图PNG`获取论文插图底图。
- 逐段`Road.route()`执行时间依赖最短路，重演边级通行时间及真实交通观测比例，与保存的`recorded_cost_sec`进行严格误差校验（0.05秒）；不一致时直接报错，不输出错误地图。
- 采用GraphML几何WKT绘制曲线路段；原图缺几何的边才退回端点直线，最后打印数量。
- 多源结果可以通过`--solutions <A> <B>`同时输入。若同一task/date/seed/solver重复出现且路线不同，优先第一份，并给出stderr警告。**不要混合不同运行配置作为同条件对照。**

## 重要实验限制

1. 此工具只用于可视化已经求解的路线，不通过视觉判定数学最优性。`solutions.json`中已有的路线及回测秒数是算法实验记录。
2. 时变道路数据只有少数路段/时段具有监测支持，未监测边仍使用静态估计。
3. 左右车辆颜色是各自方案的车辆索引，而不是按客户聚类对齐的车辆身份。观察群组变化时应结合车辆访问序列，而不是把同色线视为同一物理车辆。
4. 旧路线图的`task_0000`和v2 `tasks_10/task_0000`**不是同一组订单**，绝对不能混用任务文件。脚本会按真实道路级成本校验，但仍请检查任务版本。
5. 官方ALNS在不同运行中存在墙钟截断引起的种子重复微小波动，论文对比请采用配对均值与跨任务区间，而不是单幅地图。

### 浏览器查看

输出文件为独立单HTML。服务器不能打开浏览器时，将生成的HTML通过SCP下载到本机再用Chrome/Edge打开。

## 验证

本工具使用随项目v2求解器合成测试生成的6×6双向路网，通过4个视图、5车辆，测试道路级路段重放、边几何重建、客户覆盖、HTML/JS语法和成本一致性异常检测。**用户真实Chicago图没有在当前ChatGPT运行环境内重放**；请在服务器实际运行后核对控制台日志。运行的真实路网文件由GitHub仓库存档。
