# 记忆调用评测样例

本套件依赖同一仓库的 `go2_demo` 与 `configs`，从仓库根目录运行。它包含可离线运行的 L01–L07，以及可在真实 MuJoCo Go2 环境运行的 F01–F02。离线选择器是确定性脚本，不能证明模型理解或真实 Reader 准确率；完整评测使用项目的模型 API 和独立评分器。

| 样例 | 输入和预期 | 运行方式 |
| --- | --- | --- |
| L01 | B 的三条观测按目标与类型筛选 | 离线 |
| L02 | 干扰物与 B 分开检索 | 离线 |
| L03 | 取回路径受阻的动作反馈 | 离线 |
| L04 | 未记录目标 C，候选为空 | 离线 |
| L05 | 脚本选择最新 B 事件 | 离线；不评价语义理解 |
| L06 | 小写 b 与 B 不匹配 | 离线 |
| L07 | 无关事件不影响缓存；相关事件使缓存失效 | 离线 |
| F01 | 红色方块接近：单目标参照，独立评分验证停靠与 finish | `configs/general_agent/single_object.json`，MuJoCo |
| F02 | 先到蓝球 A 再返回遮挡后的绿方块 B，独立评分验证顺序、停靠与 finish | `configs/general_agent/occluded_return.json`，MuJoCo |

F01/F02 使用项目现有物理场景；不是新生成的障碍场景。F02 的 B 绑定和记忆查询由主模型自主决定，既不保证看见 B，也不保证查询。`query` 和 `empty` 均保留完整对话与图像，只有前者提供额外 Reader。不能将后者称作完全无记忆，也不能将一次成功归因于记忆。

## 本地离线（Windows PowerShell）

需要 Python 3.11–3.12 和仓库源码，无须 MuJoCo、模型或网络。报告写在 `.gitignore` 已排除的 `runs/` 下；若目标文件已存在会拒绝覆盖。

```powershell
cd 'D:\Academic\嵌入式系统\Agent-harness-planning-v2\Agent-harness-planning-v2'
$env:PYTHONUTF8 = '1'
python memory_benchmark/offline.py --output runs/memory-benchmark-offline-new/report.json
```

预期 `Passed 7/7`；非零退出码表示未通过。省略 `--output` 只打印结果。测试数据在 `cases.json`，其中表面点与目标身份均为合成示例，`latest` 由脚本选择。

## 完整环境（WSL / Ubuntu）

准备 Python 3.11–3.12、项目依赖、经过 `assets.lock.json` 校验的 Go2 资产、可用的 MuJoCo 渲染环境以及支持图像和工具调用的模型 API；真实实验会产生 API 费用。参考仓库根目录 README 安装依赖和资产，密钥文件置于仓库外。下方 EGL 需要本机支持；按项目文档选择合适的渲染方式。输出目录必须不存在。以下命令中的根目录和密钥路径请替换为自己的路径，所有命令在仓库根目录执行。

```bash
source .venv/bin/activate
export PYTHONUTF8=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
python scripts/fetch_assets.py --verify
python -m go2_demo.general_episode_run --task-config configs/general_agent/single_object.json --validate-only
python -m go2_demo.general_episode_run --task-config configs/general_agent/occluded_return.json --validate-only
```

可先用项目提供的真实物理接口冒烟测试，HTTP 被 MockTransport 拦截，不访问 API；它测试 `agent` 子目标更新模式且刻意提前结束，不是任务成功率测试。该检查还需要 FFmpeg：

```bash
python scripts/check_general_agent_offline.py --output runs/memory-benchmark-physical-smoke-new
```

F02 正式试点：两组使用相同模型、种子、总预算和其他参数，`empty` 无额外 Reader，`query` 允许自主查询。脚本不会读取或上传评测器真值给模型。

```bash
KEY_FILE='/path/outside/repository/key.txt'
OUT='runs/memory-benchmark-pilot-new'
python -m go2_demo.general_episode_run --task-config configs/general_agent/occluded_return.json \
  --mode empty --subgoal-updates measured --provider deepseek --model deepseek-flash \
  --api-key-file "$KEY_FILE" --assets-dir assets --seed 0 --output "$OUT/f02-s0-empty" \
  --max-requests 128 --max-reader-requests 16 --max-native-actions 460 --actions-per-slice 6 \
  --max-sim-seconds 1880 --max-wall-seconds 2400 --max-consecutive-errors 0
python -m go2_demo.general_episode_run --task-config configs/general_agent/occluded_return.json \
  --mode query --subgoal-updates measured --provider deepseek --model deepseek-flash \
  --api-key-file "$KEY_FILE" --assets-dir assets --seed 0 --output "$OUT/f02-s0-query" \
  --max-requests 128 --max-reader-requests 16 --max-native-actions 460 --actions-per-slice 6 \
  --max-sim-seconds 1880 --max-wall-seconds 2400 --max-consecutive-errors 0
python memory_benchmark/summarize.py "$OUT/f02-s0-empty" "$OUT/f02-s0-query" --output "$OUT/summary.csv"
```

F01 运行上述两条仿真命令时，将配置改为 `configs/general_agent/single_object.json`，输出目录中的 `f02` 改成 `f01`。试点后可将两组的 `--seed` 同时改为 1–4，每轮使用全新输出目录，并将全部运行目录传给汇总脚本。相同种子控制初始条件，不保证模型轨迹相同。正式评估应固定源码、模型版本与预算，保留正常失败和中断，基础设施错误单列。

## 记录和解释

`summary.csv` 按运行列出任务、组别、种子、独立成功、终止原因、`memory_query` 选择次数、查询反馈、缓存命中、分角色请求和已知 token、未知用量次数、碰撞、得分与时间。`query_choices` 不等于 Reader 请求数：空候选和缓存命中均可能不调用 Reader。缺失 token 不应当计作完整成本。若中途出错无相应字段，该字段为空；检查 `metrics.json` 与 `events.jsonl` 定位原因。

自主查询率 = 至少查询一次的 query 运行数 / query 运行数；缓存命中率 = `cache_hits / cache_observations`，分母为 0 时为 N/A。结果按全部有效运行汇总，不只看已查询或成功的运行。要计算证据准确率和召回率，需要在评测前后针对实际事件 ID 标注相关证据，再比较返回 ID；本脚本不声称能自动判断问题或 Reader 解释是否正确。

模型输入只包含公开观测；`evaluation.evaluator-only.json` 与轨迹可用于事后评分，不应加入模型输入。目标出现在图像中不等于已写入可检索的目标记录，需核查 `memory.json` 中的目标绑定、事件 ID 和 `events.jsonl`。当前实现不提供固定 A 时刻强制询问或同状态反事实重放，本套件也未新增这些功能。

## GitHub 提交范围

提交本目录的源码、样例和说明，以及运行所需的现有项目代码与配置。不要提交 API 密钥、`.env`、`assets/`、`runs/`、本机虚拟环境或本地产生的报告；根目录 `.gitignore` 已忽略常见项，提交前仍应检查暂存内容。当前套件不包含密钥、真实模型输出或数据文件。
