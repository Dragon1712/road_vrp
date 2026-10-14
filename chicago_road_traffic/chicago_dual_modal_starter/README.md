# Chicago Road + Traffic 双模态数据清洗（离线）

本脚本基于 Chicago Traffic Tracker Historical Congestion Estimates by Segment (2024-2026) 的实际 CSV 字段。脚本在服务器上不请求网络。它**只做数据清洗和张量转换**，不是交通预测模型，也没有完成 OSM 地图匹配或 TD-CVRP 求解。

## 使用

1. 在能上网的本地电脑按日期导出 `traffic_2025-06-02.csv` 等文件，再上传到服务器。
2. 将 CSV 放在 `~/road_vrp/datasets/raw/chicago/`。
3. 在离线服务器的 Python 环境中确认 `numpy` 和 `pandas` 已安装（若输出 Parquet 需安装 `pyarrow`，这些 whl 可以在有网电脑上预下载）。
4. 运行：

```bash
python prepare_chicago.py \
    --input-dir ~/road_vrp/datasets/raw/chicago \
    --pattern 'traffic_*.csv' \
    --output-dir ~/road_vrp/datasets/processed/chicago \
    --format csv
```

如本地有 `pyarrow`，可以选 `--format parquet`。请勿直接把无法上网的服务器执行 `pip install` 当作必定成功；先检查依赖是否已预装。

## 输出

- `segments_static.csv` / `.parquet`: 每个监测路段的属性及首尾坐标。
- `traffic_long.csv` / `.parquet`: 时间片、监测路段、速度（mph/kmh）、观测掩码、路段耗时秒数及 GPS 观测计数。
- `traffic_tensor.npz`: `speed_kph[T,E]`（无效/缺失值为NaN）、`observed_mask[T,E]`、`segment_ids[E]`、本地墙钟时间纳秒索引。
- `segment_index.csv`, `time_index.csv`: 张量轴对应关系。
- `daily_quality.csv`, `quality_report.json`: 每日有效率、重复数、路段属性变动及完整时间网格覆盖率。

## 重要规则

- `length` 单位 **mile**；`speed` 单位 **mph**。换算 `length_m = length * 1609.344`、`speed_kph = speed * 1.609344`、`segment_travel_sec = 3600 * length / speed`。
- `speed == -1` 表示无估计，`speed <= 0` 一律不作为可用行驶速度；保留缺失Mask。
- `direction` 如 `NB/SB` 是道路通行方向；`street_heading` 是相对城市地址网格的位置，不是通行方向。
- `bus_count` 和 `message_count` 是观测支持度；作为训练特征时要检查推理时是否真能取得未来同类信息。
- `time` 按源字符串解析成**无时区本地墙钟时间**，没有擅自解释成 UTC。`record_id` 中的时间字串不能代替官方时区规则；正式研究先核对时间戳定义，尤其跨夏令时日期。
- 当前流程按 10min 对齐，采样时刻常出现 `00:01:02`，映射到 `00:00:00` 是合理试验假设，但最好先检查完整数据的实际时间分布。`slot` 内重复记录保留最晚采样时间的一条。
- Chicago 监测道路只是部分主干道；`segments_static` **不能替代完整有向路网**。后续需要本地联网电脑另行下载 OSMnx GraphML，然后上传服务器执行道路匹配。
- 从 CSV 到 `traffic_tensor` 的道路列均按 `segment_id` 排序，使用 `segment_index.csv` 读取关系。
- 不做缺失速度填充：建议后续按训练集统计或交通预测模型填补，避免未来标签泄漏。
