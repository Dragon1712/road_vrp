# 将公开ALNS框架接入Chicago TD-CVRP30（v1兼容插件）

## 1. 上游开源项目与许可证

- 官方代码： https://github.com/N-Wouda/ALNS
- 文档：https://alns.readthedocs.io/en/latest/
- CVRP示例：https://alns.readthedocs.io/en/latest/examples/capacitated_vehicle_routing_problem.html
- 论文：N.A. Wouda, L. Lan (2023), *ALNS: a Python implementation of the adaptive large neighbourhood search metaheuristic*, JOSS 8(81): 5028. DOI: https://doi.org/10.21105/joss.05028
- 许可证：MIT，原始版权归Niels Wouda和贡献者。本文新写的适配器不包含上游库的源代码复制，**正式运行必须额外安装官方`alns==7.0.0`**。如果将上游源码复制、改写或打包分发，必须保留原始MIT许可证和版权声明。

**选择原因**：官方库已实现核心ALNS迭代、轮盘赌/均匀算子选择、退火接受、停止条件和搜索统计。我们只开发真实道路时变VRP的自定义状态与算子；沿用你的`road_eval.py`、`models.py`和`benchmark_suite.py`。旧版自写ALNS保持不变，可作为内部对照。

## 2. 文件与安装

从GitHub已有仓库`~/code/WZL/road_vrp`根目录执行：

```bash
unzip chicago_upstream_alns_bridge.zip
```

ZIP仅含以下**新增**文件，不覆盖已存在的v2代码：

```
chicago_tdcvrp_solver_suite_v2/
  upstream_alns_adapter.py
  benchmark_official_alns.py
  run_official_alns.sh
  requirements_upstream_alns.txt
  tests/test_upstream_bridge.py
  compare_official_alns.py
README_使用前先读.md
```

在**联网且与服务器CPU架构、Python版本、Linux平台相匹配的机器**下载官方库及所有依赖（尤其NumPy和matplotlib的Linux wheel）：

```bash
python -m pip download -d alns_wheelhouse 'alns==7.0.0'
```

上传`alns_wheelhouse/`到离线Ubuntu服务器。激活原有环境后离线安装：

```bash
conda activate WZLroadvrp
python -m pip install --no-index --find-links ./alns_wheelhouse 'alns==7.0.0'
python -c 'import alns; from alns import ALNS; print("ALNS import succeeded")'
```

如果使用pip虚拟环境而不是conda，按实际环境激活即可。上游库依赖NumPy、matplotlib，切勿盲目用Windows wheel安装在Ubuntu上。服务器无法联网时不要执行普通`pip install alns`。

## 3. 先做合成测试（官方包安装后执行）

```bash
cd chicago_tdcvrp_solver_suite_v2
python -m tests.test_upstream_bridge
cd ..
```

测试涵盖6个外部ALNS变体的状态不可变、客户唯一覆盖、车辆容量、时变成本计算、静态/动态模式、完整benchmark输出格式。

**诚实说明**：本插件在交付前的执行环境中没有安装上游`alns`库。因此完成了合成图的自定义算子测试与**官方API形式的测试替身**集成测试，但还没有成功在本环境对官方7.0.0运行时进行验证。测试日志如果写了`TEST DOUBLE`表示不是官方包测试；请在服务器安装官方包后重跑。

## 4. 使用本仓库真实道路成本运行

```bash
cd ~/code/WZL/road_vrp
bash chicago_tdcvrp_solver_suite_v2/run_official_alns.sh smoke
```

通过后，再运行：

```bash
bash chicago_tdcvrp_solver_suite_v2/run_official_alns.sh compare
bash chicago_tdcvrp_solver_suite_v2/run_official_alns.sh ablations
```

两个正式命令分别使用 `10任务×6情景×3种子`：
- `compare`：已有自写ALNS `alns` vs 官方引擎适配 `upstream_alns`，各180条结果。
- `ablations`：上游引擎Full与5个模块消融，共1080条结果。

正式对比还会生成`comparison/paired_results.csv`、`task_means.csv`、`solver_comparison.csv`，以相同任务/日期/种子计算配对胜率和任务区组Bootstrap区间。

`compare`和`ablations`都只有`oracle`规划输入；是离线Teacher构造/方法验证，不可宣称真实未来交通预测。要比较相对于静态规划的收益，请在独立新输出目录对两种`--planners static,oracle`运行完全同样的设置。

如果中断，在相同代码、任务、成本、参数条件下加`RESUME=1`：

```bash
RESUME=1 bash chicago_tdcvrp_solver_suite_v2/run_official_alns.sh compare
```

不要直接覆盖以前的`runs/chicago_solver_suite_v2/base_10tasks`、`ablations_10tasks`。本插件输出统一放在`runs/chicago_solver_suite_v2/upstream_library/`。

## 5. 开源框架与本研究模块分工

| 组件 | 实现来源 | 说明 |
|---|---|---|
| ALNS主循环/算子选择/退火接受 | 官方`N-Wouda/ALNS` | 不自行重复实现，便于方法学审查 |
| CVRPState / 破坏和修复封装 | 此插件`upstream_alns_adapter.py` | 不可变状态，破坏后修复，错误修复回退 |
| random/worst/related + greedy/regret2 | 现有v2`solvers.py` | 直接复用，确保只对比ALNS框架而非换数据或换目标 |
| 本地搜索 | 现有v2`first_improvement` | 每5个修复候选执行最多60次，支持关闭消融 |
| 时变OD与静态OD | 现有v2`road_eval.ODCost` | 20分钟时变OD到达函数插值 |
| 精确评估 | 现有v2`Road.exact` | 逐段时变最短路，保留Chicago实测掩码 |
| 实验配置、缓存、断点续跑 | 现有v2`benchmark_suite.py` | 不更改原始代码，动态注入新求解器方法 |

**重要：** 代码中采用独立的`CombinedStop`同时限制最多迭代和搜索墙钟时间。单步操作可能超时，因此严格时间控制也需要在实验日志记录超时量。ALNS退火按照`max_iterations`预定进度冷却；不同时间预算可能使实际迭代数不同，研究公平性时应额外采用固定迭代对照并报告实际执行的迭代次数/评价次数。

## 6. 六个变体及意义

- `upstream_alns`：3种破坏、2种修复、自适应RouletteWheel、SA接受及局部搜索。
- `upstream_no_related`：去相关性破坏。
- `upstream_no_worst`：去Worst Removal。
- `upstream_greedy_repair`：仅保留贪心修复。
- `upstream_uniform`：随机等概率算子选择。
- `upstream_no_local`：关闭局部搜索。

`--methods`中允许同时包含`alns`、`sa`、`vnd`、`upstream_alns`等，不会破坏原有方法。统一记录`surrogate_travel_sec`、`ex_post_road_travel_sec`、`score_calls`、`vehicle_loads`、`termination_reason`等指标。结果中增加了各算子使用次数和ALNS官方结果的Outcome计数，保存在 `solutions.json -> solver_diagnostics`。不得把不同算法的单次迭代视为等价的工作量。

## 7. 复现核验与研究限制

1. `benchmark_suite.py`不修改；全套仍使用同一`edge_costs.npz`和相同的10个task JSON。
2. 当前GitHub的 `tasks_10` 若尚未纳入仓库，需要先同步，因为正式结果只能靠准确任务实例复现。
3. 服务器单次搜索受时钟波动影响；至少3个随机种子，报告均值/配对胜率/10任务区组Bootstrap区间，不只报告最好值。
4. `oracle`为事后历史交通，不是未来预测。没有引入神经网络，不能宣称多模态学习收益。
5. 本包没有复制上游ALNS源码，因此所有新增代码都可由git单独审查；建议仓库新分支合并之前先在服务器验证依赖与测试。
6. 正式论文仍建议额外引入强独立基线（例如PyVRP静态CVRP）和小实例的精确/下界验证，不把本库结果称作最优。
