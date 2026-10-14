# Chicago TD-CVRP30：五车道路级路线可视化（完全离线）

## 输出内容

`visualize_tdcvrp_routes.py` 从已有 VRP 方案恢复 **30 个客户、5 辆车在 OSM 有向图上的真实形状道路路径**，生成一个**单文件、完全离线的交互式 HTML**。背景是 GraphML 文件中的 OSM 路网；**无线上地图瓦片、无 CDN、无外网请求**。道路几何按 GraphML 的 `geometry` WKT 显示；少数没有 geometry 的边画成节点连接的直线。

页面可以：

- 切换 `2025-06-05_0800` / `2025-06-06_0800` 等交通日期。
- 切换 `static` / `oracle` 两类**客户分组与访问顺序方案**。
- 同时显示5辆车的彩色路径，或单独勾选车辆；显示仓库 D、编号客户点、客户访问顺序、车辆载重与行驶时间。
- 鼠标悬停客户查看车辆、配送次序、需求与到达时刻；滚轮放缩、拖拽平移。
- 查看同一情景下静态与Oracle解的行驶时间差异。

**方法界限（重要）**：彩色道路线不是GPS行驶轨迹，而是依据已记录的客户顺序，在该日期的历史交通成本下，逐OD重新计算的时变最短路径。静态方案也按同样规则道路级重路由，公平对比的是客户分组和访问顺序；**不是**把静态道路路径固定住不动。历史拥堵数据事后已知，此处不是预测模型效果。未观测边/时段仍由静态估计边权回退。

## 一、使用当前服务器目录运行

前提：之前的 TD-CVRP30 程序可运行，并已安装 `numpy pandas networkx`；无需安装 Folium、Plotly 或连接外网。

将ZIP解压到项目根目录：

```bash
cd ~/code/WZL/road_vrp
unzip chicago_route_map_offline.zip
```

直接使用你上传的 seed_42 求解结果：

```bash
python chicago_route_map_offline/visualize_tdcvrp_routes.py \
  --solutions runs/chicago_tdcvrp30/multiseed/seed_42/solutions.json \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --output runs/chicago_tdcvrp30/maps/routes_seed42.html
```

文件会包括 `task_0000` 的4个视图：6月5日/6日×静态/动态；默认显示6月5日 Oracle 配送解（车辆用5种不同颜色）。

如果想看其他种子，把 `seed_42` 换成 `seed_43` 等，并改变 `--output` 文件名。例如：

```bash
python chicago_route_map_offline/visualize_tdcvrp_routes.py \
  --solutions runs/chicago_tdcvrp30/multiseed/seed_43/solutions.json \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --output runs/chicago_tdcvrp30/maps/routes_seed43.html
```

只画6月5日动态方案：

```bash
python chicago_route_map_offline/visualize_tdcvrp_routes.py \
  --solutions runs/chicago_tdcvrp30/multiseed/seed_42/solutions.json \
  --task-dir datasets/processed/chicago/tdcvrp30/tasks \
  --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
  --cost-dir datasets/processed/chicago/dynamic_costs \
  --scenario 2025-06-05_0800 --planner oracle \
  --output runs/chicago_tdcvrp30/maps/oracle_0605.html
```

## 二、怎样查看地图

服务器可断网生成 HTML。若服务器仅有SSH命令行，建议**把 HTML 下载到本地电脑，用Chrome/Edge/Firefox打开**：

```bash
# 在你有访问服务器权限的本地电脑执行，而非服务器上
scp xys@<SERVER_IP>:~/code/WZL/road_vrp/runs/chicago_tdcvrp30/maps/routes_seed42.html .
```

打开 `routes_seed42.html`，不需要互联网，灰色道路背景来自你保存的OSM GraphML本身，而非OpenStreetMap在线瓦片。

## 三、报错如何排查

- `No module named 'tdcvrp'`：请在包含 `tdcvrp/road.py` 的项目根目录执行，且确保目录未移动。
- `FileNotFoundError: task_0000.json`：使用之前生成任务时的原目录，不能重新生成不同种子的30客户任务覆盖它。
- `N edge IDs ... missing in GraphML`：`chicago_loop_drive.graphml` 与生成 `edge_costs.npz` 时使用的不是同一版本，请恢复同版本文件。
- `Road-level recomputation differs from solutions.json`：通常是成本张量版本、时间索引或任务文件与求解时不一致。为避免绘制不可信的道路路径，程序会报错停下而非继续画。
- 浏览器无图：应先检查生成过程是否报错，HTML文件是否完整。地图完全离线，不依赖本地电脑联网。
- 图中道路看起来不是商业底图：这是预期行为。我们画的是GraphML真实道路几何，避免在线瓦片导致服务器断网无法查看。

## 四、科研图建议

第一张展示 **2025-06-05 08:00 Oracle** 五车道路轨迹，每辆车一种颜色；第二张同一情景 Static，保持相同地图缩放；第三张展示 **2025-06-06 08:00 Oracle**。注明车辆行驶时间含实测监测走廊的动态标签，也含大量静态估计路段。地图本身不能单独证明对比学习有效。

## 五、已完成的测试

脚本已在小型合成有向路网上跑通：4种方案视图、逐段时变最短路与已有解的精确时间成本一致性校验、HTML打包生成。真实Chicago GraphML尚未在本对话上传，因此真实地图必须在你服务器上生成，或上传真实GraphML/任务/动态成本文件以便这里生成。不要将合成测试地图当成Chicago真实路线。
