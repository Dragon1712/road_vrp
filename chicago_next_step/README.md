# Chicago Road + Traffic: 离线OSM匹配（候选版）

本工具包专门对应 `segment_coverage.csv` 的实际15列，以及 Chicago Traffic Tracker 2025数据。**用途仅是将监测走廊候选匹配到OSM有向道路路径，并输出可审核的匹配表**；它尚未将时变速度广播到OSM每条边，也不包含TD-Dijkstra或TD-CVRP求解器。

## 1. 本地有网络的电脑：下载OSM

Python>=3.10，推荐 OSMnx>=2.0。在本地安装：

```bash
python -m pip install 'osmnx>=2.0,<3'
python download_osm_locally.py --output chicago_loop_drive.graphml
```

默认下载矩形：W -87.667, S 41.851, E -87.592, N 41.906。需要时可用 `--west`, `--south`, `--east`, `--north` 改动。此命令**需要本地电脑能连接OpenStreetMap的Overpass服务器**。若访问失败，可通过Geofabrik等合法OSM数据包并在本地裁剪导出GraphML，之后的服务器匹配不联网。

复制 GraphML 文件到离线Ubuntu服务器。

## 2. Ubuntu离线服务器：运行候选匹配

所需依赖：`pandas`, `numpy`, `networkx` （需要提前安装或离线准备wheel）。不需要OSMnx和网络。

```bash
python match_segments_offline.py \
 --segments datasets/processed/chicago/qc/segment_coverage.csv \
 --graphml datasets/raw/chicago/chicago_loop_drive.graphml \
 --output-dir datasets/processed/chicago/osm_match
```

输出 `segment_matches.csv` / `accepted_osm_edge_mapping.csv`。其中 `match_status==accepted` 只表示通过**初步**阈值校验；后续必须地图抽查和方向检查。筛选有良好速度监督的路段在统计分析阶段做，不要直接删除完整道路图里的低覆盖监测路段。可用 `--min-coverage 0.5` 仅做高质量匹配候选，但最终完整OSM网络仍应保留。

## 3. 匹配逻辑与局限

- 使用监测路段的起点和终点坐标近邻匹配到OSM节点（默认175m以内）。
- 在**有向**OSM路网上寻找匹配街道名称的偏好路径，允许跨越多条OSM边。
- 用道路名匹配路径长度占比（默认>=50%）与原始长度比（默认0.65–1.85）筛掉明显错误匹配。
- 这是一种可复现的初筛实现，并非可直接发表的地图匹配真值。街道并行、辅路、转向限制、OSM简化节点、端点偏移可能导致错误；路网不一定编码所有真实转向限制。后续建议验证WKT折线路径/方向，或调用本地OSRM/Valhalla进行高精度匹配。
- 若多个Chicago segment落在同一OSM边，后续融合动态边权必须建立冲突策略、保留映射置信度，不能直接后者覆盖前者。
- 新生成文件不包含未来真实速度；推理时需从模型预测的未来边权产生TD-OD。

## 4. 下一步检查

优先汇报：`Accepted / all`，`Accepted among >=50% coverage`，方向不一致率，路径长度比值，以及通过地图人工审核的20–50段准确率。
