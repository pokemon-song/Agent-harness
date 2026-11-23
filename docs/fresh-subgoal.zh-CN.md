# A/B 遮挡返回：持续子目标版本 v1

当前复现配置是 **持续子目标规划＋A→B 完整上下文保留＋碰撞扣分后继续**。任务先到蓝球 A，再返回绿方块 B；两组各从仿真起点独立开始。最新一次对照的有记忆组完成任务，但没有查询外部记忆，不能据此证明记忆 Agent 带来了收益，详见下方结果。

此版本借鉴 [Agentic Robot](https://arxiv.org/abs/2505.23450) 的子目标执行、检查与恢复流程。规划模型自主提出移动目标；程序保存执行状态、验证公开观测中的进展。取消新版本中的候选观察位置生成和 `explore_viewpoint` 工具。

这是本项目的导航适配：执行器沿用 RGB-D 检查与 Go2 运动控制，检查器采用可审计的公开观测规则。没有移植原论文的 OpenVLA 机械臂策略或训练视觉验证器；单轮结果不能代表稳定成功率。

## 任务与判定

当前示例使用 24×18 米场地，A 为蓝球（6.4, 0），B 为绿方块（1.8, 0.6），柜体遮挡两者之间的视线。目标真值坐标用于场景和独立评分，不直接提供给规划 Agent；规划 Agent 获取当前图像、公开 RGB-D 候选和里程计。

A、B 分别要求机器人距目标中心 0.82–1.18 米、朝向误差不超过 0.25 rad，并确认停稳。A 合格后才切换到 B；最后须调用 `finish`，并由独立评分再次验证 B。碰撞扣分但不阻止目标通过，跌倒仍判失败。

## 决策流程

1. 规划 Agent 读取当前图像、RGB-D 目标候选、里程计、深度摘要、最近六条物理动作反馈，以及持续保存的 `task_state`。
2. 用 `set_subgoal` 建立一个近期目标；需要过去信息时，照常调用按需记忆查询。
3. 模型选择数值机身路点或转向，每次提供简短的 `intention`，说明要取得的进展。
4. 执行器最多执行 6 个原生动作，然后返回当前观测；原生动作包括移动、转向、跟踪测量、停止等。发现当前阶段目标也可提前中断观察路线。
5. 检查器更新已完成路点、子目标状态、最后目标目击、实际运动和受阻原因。模型据此继续原目标、采取恢复动作，或明确说明原因后更换目标。

`set_subgoal` 有三种检查条件：

| criterion | 所需参数 | 完成依据 |
| --- | --- | --- |
| `reach_pose` | 自主选择的 `target_pose={x,y,yaw?}` | 实测里程计距该机身点 ≤0.19 米；指定朝向时误差 ≤0.12 rad |
| `observe_target` | 描述要获取的目标观测 | 当前 RGB-D 候选中出现当前阶段目标 |
| `dock_target` | 描述停靠当前阶段目标 | A 使用独立合格停止事件；B 记录控制器完成声明，最终通过仍由独立评分确认 |

目标标签由当前 A/B 阶段确定。模型不能通过子目标文本、自报完成或历史目击替代实际停靠。上表中机身路点的误差阈值与任务的目标停靠距离是不同概念。

## 持续进度和恢复

`task_state` 保存当前子目标、最近四个结束或替换的子目标、最近目标目击、最近一次运动意图，以及至多 32 个模型选择的路线点。状态大小不随总决策次数持续增长。

`follow_waypoints` 新增 `intention` 和 `route_role`：

- `primary`：提交或明确替换当前目标的主路线。执行器实测完成的路点前缀会被删除，`continue_subgoal` 续走剩余部分。
- `recovery`：执行模型选择的局部调整，保留原目标和未完成的主路线。

路线位置已完成、末端转向或停靠未完成时，单独保留末端操作。续停靠从当前机身位置开始，不返回停靠启动时的旧路点。

深度拒绝、执行不收敛、连续三次没有足够实测进展，或近期反复回到近似位置和朝向，会触发 `recovery_required`。恢复期关闭自动续走入口；模型可重新选择短程移动或扫描，执行方向由模型决定。三次无进展是启发式提醒，不能保证已经完整识别所有失败。

查询记忆不计入物理停滞次数，也不抹掉当前失败状态。目标暂时离开画面时保留最后目击，并标明图像坐标属于当时视角。

例如：返程重新看到 B 后，设立 `dock_target`。直接靠近被柜角阻挡时，该目标仍保留；模型可以提交 `route_role=recovery` 的绕行短段，然后根据当前画面重新接近 B。

## 消融边界和记录

`empty`、`ledger`、`query` 使用相同的规划提示、子目标状态、检查器和底层动作。`empty` 不提供额外文档记忆，`ledger` 用确定性规则读取记录，`query` 由记忆读者按请求选择已有证据。`empty` 也有本阶段执行状态；消融比较的是额外文档记忆是否可用，而非完全抹掉短期任务进度。

默认 `--handoff-context reset`：A 合格停靠后，三组均重建 B 阶段的子目标状态，清空 A 阶段动作历史、最后目击、待执行路线及失败记录。有记忆组通过既有记忆接口获取本轮冻结的去程文档。

`--handoff-context retain`：两组都保留完整的公开规划对话，包括此前图像、模型工具选择及执行反馈；有记忆组还保留已有记忆答复卡、查询反馈和缓存。A 的最终执行状态保存在 B 状态的 `previous_stage_state` 中。当前任务明确切换到 B，先前 A 的待执行动作不自动成为 B 的动作。无记忆组仍不启用额外文档记忆，但可直接利用去程对话。

retain 模式测试的是保留正常上下文时额外记忆模块的作用。它与清空去程上下文的 reset 模式属于不同实验条件，应分别报告。完整对话随回合累积，输入开销也会增加。

当前碰撞规则为 `record_and_penalize`：B、柜体、边界等场景接触均记录并扣分，继续执行，不因接触直接判定目标失败。跌倒和预算限制仍按原规则结束。A、B 仍须满足独立判定的距离、朝向和停稳要求，整轮还须执行 `finish`；`finish` 本身不是通过证明。

A 通过得 50 分，B 在整轮成功时再得 50 分；每个独立碰撞事件扣 5 分，总分最低为 0。事件次数与接触持续的物理步数不同。是否通过与扣分后的总分分别记录，即使发生碰撞，只要满足目标完成条件仍可通过。碰撞单价和目标分值以当轮 `manifest.json` 为准。历史 r1–r3 等已完成实验按当时清单中的碰撞规则解释，不回写为新标准。

每轮仍保存完整录像、逐步画面、轨迹和指标；`events.jsonl` 新增 `subgoal_set`、`subgoal_verification`，规划输入记录 `task_state`，各阶段指标保留最终状态。运行清单记录 `persistent_subgoals_v1`、检查器类型、分段上限和相关源码摘要。

## 运行当前两组对照

获取本次发布的 `planning-v1` 分支：

```bash
git clone --branch planning-v1 --single-branch \
  https://github.com/pokemon-song/Agent-harness.git
cd Agent-harness
```

先按 [README 的安装步骤](../README.md)安装 Python 依赖，并执行 `python scripts/fetch_assets.py` 获取校验过的资产。在仓库根目录运行以下两条命令，使用同一个新的输出根目录。凭据文件 `/path/to/key.txt` 由运行者自行提供，不包含在仓库中。

此实验入口固定使用 `deepseek-flash`、关闭思考，每次最多输出 8192 token，API 超时 90 秒。下列命令会产生 API 费用；请求、原生动作、仿真时间和墙钟预算均按 A/B 阶段分别计算。1024 次请求包括记忆读者请求，连续规划错误上限为 6。

```bash
# 无额外记忆，保留正常公开对话
MUJOCO_GL=egl python -m go2_demo.fresh_episode_run \
  --navigation subgoal --handoff-context retain --mode empty --seed 0 \
  --floor-size-m 24 18 \
  --max-requests 1024 --max-native-actions 230 \
  --max-sim-seconds 940 --max-wall-seconds 2234 \
  --assets-dir assets --api-key-file /path/to/key.txt \
  --runs-dir runs/fresh-subgoal-retain-s0-new1

# 按需文档记忆，同样保留正常公开对话
MUJOCO_GL=egl python -m go2_demo.fresh_episode_run \
  --navigation subgoal --handoff-context retain --mode query --seed 0 \
  --floor-size-m 24 18 \
  --max-requests 1024 --max-native-actions 230 \
  --max-sim-seconds 940 --max-wall-seconds 2234 \
  --assets-dir assets --api-key-file /path/to/key.txt \
  --runs-dir runs/fresh-subgoal-retain-s0-new1
```

两条命令分别写入 `empty/` 与 `query/`，每组都重新初始化仿真、对话和记忆，不读取另一组的记录。重跑时换一个新的输出根目录；已有同名组目录会拒绝覆盖。固定仿真 seed 不保证模型输出或任务成绩相同。

请保留显式的 `--handoff-context retain`：程序默认仍为 `reset`，省略会清空去程规划上下文。`--mode all` 会运行 `empty`、`ledger`、`query` 三组，不等于上述两组对照。旧观察位置方案须使用 `--navigation viewpoints --handoff-context reset` 和新输出目录。

运行结束后保存 `manifest.json`、`metrics.json`、`events.jsonl`、`trajectory.jsonl`、逐步图像及 `video/` 里的模型和位姿记录；有记忆组另有 `episode_documents.json`。归档脚本从记录生成 MP4，并核验与打包，无须再次调用模型。需系统 `ffmpeg`，或先安装 `python -m pip install imageio-ffmpeg` 作为备用编码器：

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/archive_fresh_episode.py \
  runs/fresh-subgoal-retain-s0-new1 --modes empty query
```

默认输出到 `results/fresh-subgoal-retain-s0-new1/`，包含每组全程录像、进入 B 时的返程录像、审计结果、汇总和数据包。录像按仿真时间组织，不含模型请求时暂停仿真的等待时间。

归档会根据运行清单中的 `handoff_context` 核验对应交接行为。retain 模式校验历史消息计数没有下降、第一条 B 输入仍保留 A 的最终状态及最后记忆答复；reset 模式继续要求 B 初始上下文为空。新碰撞规则下还核验接触事件数、目标分、扣分及总分，并将 `goal_points`、`collision_count`、`collision_penalty`、`score` 写入归档汇总；历史没有这些字段的实验仍可归档。

## 最近一次真实运行

2026-11-23，`fresh-subgoal-retain-24x18-s0-1024-r4`，使用上述配置，每组各一个样本：

| 指标 | `empty` 无额外记忆 | `query` 按需记忆 |
| --- | ---: | ---: |
| A / B | A 未通过，未进入 B | A、B 均通过 |
| 整轮结果 | 主动放弃 | `finish`，独立判定通过 |
| 规划请求 | 49 | 93（A 60，B 33） |
| 记忆查询 / 读者请求 | 0 / 0 | 0 / 0 |
| 碰撞事件 | 2 | 1 |
| 目标分 / 扣分 / 最终分 | 0 / 10 / 0 | 100 / 5 / 95 |

有记忆组写入了经历，但未调用 `memory_query`；它可以利用保留的去程对话。因此本轮验证了当前规划闭环能完成任务及碰撞后继续的行为，不能证明外部记忆检索提升了成绩。

- [完整实验报告](../results/fresh-subgoal-retain-24x18-s0-1024-r4/report.zh-CN.md)
- [机器可读汇总](../results/fresh-subgoal-retain-24x18-s0-1024-r4/summary.json)
- [有记忆组全程录像：A/B 标注、停稳段压缩，8 分 55 秒](../results/fresh-subgoal-retain-24x18-s0-1024-r4/query/full-labeled-compressed.mp4)
- [该录像的帧映射和摘要](../results/fresh-subgoal-retain-24x18-s0-1024-r4/query/full-labeled-compressed.json)
- [无额外记忆组全程录像](../results/fresh-subgoal-retain-24x18-s0-1024-r4/empty/full.mp4)
- [逐步实验数据包](../results/fresh-subgoal-retain-24x18-s0-1024-r4/episode-data.zip)

标注录像从同一次物理记录重新渲染，保留全部去程与返程，只压缩停稳片段。目标标签和轨迹线仅供观看，不是规划 Agent 的输入；旧导出因远端裁剪导致 A/B 难以看见，新导出已修正。

数据包包含日志、轨迹、位姿采样、指标与记忆记录，不含原始观测图像或 `video/model.mjb`。它可用于核验动作和评分，不能独立重渲染本轮录像。

`export_labeled_episode_video.py` 是本次 r4 有记忆组的专用展示脚本：使用已保存的 `full-pauses-compressed.json` 帧映射、固定镜头距离和“有记忆组”标题，不自动为任意新实验计算停稳压缩时间线。重新导出需要保留该轮原始 `runs/.../query/`（包括 `video/model.mjb`）、MuJoCo 3.2.5、上述 ffmpeg 编码器，以及 Ubuntu 的 `fonts-noto-cjk` 字体包；脚本固定读取 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` 和 `NotoSansCJK-Bold.ttc`。新实验请使用上面的通用归档命令。

## 实现入口与离线检查

| 文件 | 作用 |
| --- | --- |
| [fresh_episode_run.py](../go2_demo/fresh_episode_run.py) | 独立回合、阶段交接、预算和碰撞评分 |
| [fresh_subgoal_planning.py](../go2_demo/fresh_subgoal_planning.py) | 规划工具协议与完整公开对话保留 |
| [fresh_subgoals.py](../go2_demo/fresh_subgoals.py) | 子目标进度、路点续走与恢复状态 |
| [fresh_episode_memory.py](../go2_demo/fresh_episode_memory.py) | 文档记录、按需证据选择与记忆卡 |
| [archive_fresh_episode.py](../scripts/archive_fresh_episode.py) | 录像归档、交接和评分审计 |
| [export_labeled_episode_video.py](../scripts/export_labeled_episode_video.py) | 从 r4 原始记录及已有帧映射导出专用标注录像 |

本地离线检查：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

协议测试使用模拟 API，流程测试覆盖阶段重置、路线分段续走、受阻后的记忆查询与恢复，以及停靠中途续执行。离线测试通过不等于真实模型 A/B 任务通过。

本次发布前的完整离线验证：**576 通过、15 跳过**（2026-11-23，EGL，23.23 秒）。覆盖子目标检查、分段续走、完整上下文交接、跨阶段动作 ID、碰撞后继续与独立扣分，以及实验归档审计。历史 reset 模式、碰撞即结束模式属于不同实验条件，不与当前 retain 结果混算。
