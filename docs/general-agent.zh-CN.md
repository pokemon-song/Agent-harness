# 通用子目标与按需记忆版本

## 程序依据实测结果更新子任务（当前默认）

`general_episode_run` 默认恢复为 `--subgoal-updates measured`。主模型仍选择子任务、目标像素、路线和恢复动作；程序依据当前有效目标绑定、实测位姿及技能执行反馈判断局部子任务完成，推进已完成路点，并在受阻或连续无进展时标记需要恢复。程序不会替模型选择下一目标、观察位置或绕行路线。独立任务评分不读取子任务完成标记，局部技能成功仍不等于 A/B 任务通过。

直接构造 `GeneralAgent` / `SubgoalState` 同样默认 `measured`。完整上下文、按需记忆、底层执行与评分不变，连续错误次数上限仍默认关闭。历史实验的冻结源码和记录不受本次默认模式恢复影响。

## Agent 自主更新子任务（可选实验模式）

显式使用 `--subgoal-updates agent` 可复现自主判断版本。主决策模型通过 `update_subgoal` 显式判断当前子任务的 `active`、`recovering`、`completed` 或 `abandoned` 状态，并给出简短依据。此模式中程序记录当前位姿、目标绑定、动作实际结果与无进展诊断，但这些事实不自动完成子任务、推进保存的路线或触发恢复状态。

更新必须指定当前 `goal_id`。可选 `completed_waypoints` 表示**当前主路线累计**已完成的前缀数量；不是最近一次执行分片内的数量。`route_done=true` 表示模型确认整条路线及末端朝向已完成，清空保存路线。完成或放弃子任务也会清空其路线。程序校验 ID、枚举、数量范围与单调性；判断真实性仍由模型负责，判断记录标为 `agent_judgment`。

底层几何跟踪、到达测量、运动控制、动作分片、深度检查、预算和跌倒终止保持原逻辑。独立任务评分不读取模型的子任务完成声明；`update_subgoal(status="completed")` 不会使 A/B 获得分数。状态更新调用使用主模型本身，没有新增验证模型 API；它消耗主模型请求，但不消耗物理动作。

下面的架构与工具说明以当前默认的 `measured` 为准；模型显式更新行为仅属于可选的 `agent` 模式。

本版本保留“主决策模型 → 持续子目标/路线 → 分段执行 → 反馈验证 → 主模型调整”的结构，并保留独立的按需记忆读者。通用核心不再包含目标 A/B、颜色检测、固定先后阶段、绕柜提示、预设观察位置或返程专用检索类别。

这是新入口 `go2_demo.general_episode_run`。早期 `fresh_episode_run` 和冻结实验结果保留，供历史复现；新旧版本的结果不能视为只改变了一个变量。

## 职责与信息边界

| 部分 | 实现 | 是否调用大模型 API |
| --- | --- | --- |
| 主决策 | 根据任务描述、当前 RGB、公开位姿和反馈，设置子目标、选择路点及恢复动作 | 是 |
| 目标定位 | 主模型在当前图像指定目标 ID 和像素；本地 RGB-D 反投影与光流追踪 | 指定像素是主模型决策；几何计算不是 |
| 子目标状态 | 程序保存目标与路线，依据实测结果自动更新局部完成、路线进度及恢复标记 | 否；下一目标与恢复动作仍由主模型选择 |
| 路线执行 | 沿模型指定路点闭环移动，当前深度检查阻挡，每次有限动作后返回 | 否 |
| 接近技能 | 根据当前有效追踪点调整距离、朝向、停稳；失跟后返回重定位 | 否 |
| 记忆写入 | 全程保存公开观测和动作请求/实测反馈，保留来源区别 | 否 |
| 记忆读取 | 按问题与可选事件类型/目标 ID 筛选，Reader 引用已有事件编号 | Reader 是；范围过滤、缓存和原文回填不是 |
| 任务适配与评分 | 提供任务描述、公开完成条件及验收布尔反馈，独立判定最终结果 | 否 |
| 物理控制 | 原有本地预训练步态网络、关节控制与 MuJoCo | 本地神经网络推理，不调用远程 API |

`general_agent.py`、`general_subgoals.py` 和 `general_memory.py` 不导入场景地图、任务颜色检测器或评分器。`general_execution.py` 复用通用 Harness、点追踪和深度检查。只有 `general_episode_run.py` 中的环境适配器访问场景定义和独立评分；向 Agent 提供的验收信息仅包含公开目标 ID 和布尔值，不包含真实目标坐标、距离误差或评分器内部状态。

## 保留的结构

- 一个主模型负责子目标、路线、恢复和记忆查询，没有额外独立规划 LLM。
- 一个当前子目标，可选择 `reach_pose`、`observe_target` 或 `approach_target`；目标由显式 `target_id` 指定。
- `primary` 保存/替换主路线，`recovery` 保留主路线；程序依据实际完成情况裁剪已完成前缀，`continue_subgoal` 续走其余部分。
- 未完成目标替换需要理由；程序依据受阻、重复视角、连续无进展标记需要恢复，恢复动作和新路线仍由模型选择，不自动生成绕行点。
- 每次执行分片有限，物理预算和原生动作编号在整轮累计。
- 主模型保留整轮对话、图像和工具反馈，没有按任务里程碑清空上下文。
- Reader 独立按需调用，原始证据回填，查询结果可缓存；新匹配事件使缓存失效。

## 去除或替换的任务特化

1. 蓝/绿颜色阈值目标检测改为 `ground_target(target_id, image_id, u, v)`。目标语义由主模型判断；程序只证明该像素有有效深度和可追踪几何。
2. 主循环没有 A/B 阶段分支，不自动切换阶段、冻结去程记忆或加载返程提示词。任务顺序由配置里的任务要求表达，由环境评分验证。
3. 不再有 `a_sighting`、`b_sighting`、`route_reverse` 等固定检索意图。通用事件库按任意目标 ID 和事件类型筛选，Reader 根据问题选证据。
4. 接近技能不自动寻找某种颜色，也不自动调用 `finish`。丢失目标时返回模型重新指定当前像素。
5. 任务目标描述、顺序、完成标准和环境场景选择独立配置；配置没有给 Agent 隐藏目标坐标或绕行路线。

## 几何成功不等于任务成功

`ground_target` 的成功仅表示模型指定的点有可用几何。模型若把地板认作目标，程序可能仍可追踪该点，因此反馈明确标注 `identity_basis=model_proposed` 和 `geometry_only=true`。

`observe_target` 只验证当前帧中存在对应的有效绑定，不代表程序独立完成了语义识别。

任务配置最多可声明 32 个目标，但复用的点追踪器同时只支持 8 个活跃点；这是局部几何容量，不是历史事件库的容量。重新指定同一目标时会释放其旧绑定点，避免反复重定位耗尽槽位。

`approach_target` 使用**所选可见表面点**的距离，返回 `skill_completed` 只说明局部几何与停稳条件满足。任务配置若要求物体中心距离，仍以环境验收为准，主模型需要继续调整。不能把表面距离和中心距离混作一种测量，也不能把 `finish` 或子目标完成标记当作整轮通过。

定位、深度和边界保护仍来自原仿真设施。本次移除的是任务特定决策规则，不是重新实现真实机器人 SLAM、通用语义分割或定位噪声模型。边界保护拒绝越界动作，碰撞本身记录扣分后继续；跌倒仍会终止。

## 配置和运行

提供两个使用同一核心的任务配置：

- `configs/general_agent/occluded_return.json`：原 A/B 遮挡任务。A/B 只出现在任务数据和旧评分器的检查点映射中。
- `configs/general_agent/single_object.json`：单目标接近，使用不同目标描述和任务流程。

在仓库根目录验证配置，不启动仿真、不调用 API：

```bash
python -m go2_demo.general_episode_run \
  --task-config configs/general_agent/occluded_return.json --validate-only
```

运行新的有按需记忆版本（这条命令会调用模型 API，输出目录必须不存在）：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python -m go2_demo.general_episode_run \
  --task-config configs/general_agent/occluded_return.json \
  --subgoal-updates measured \
  --mode query --provider deepseek --model deepseek-flash \
  --api-key-file /path/to/key.txt --assets-dir assets \
  --output runs/general-occluded-return-new
```

对照组只改 `--mode empty` 并使用另一全新输出目录。两组均保留完整上下文；`empty` 表示无额外 Reader，不表示模型没有上下文记忆。

默认整轮主模型和 Reader 共用 128 次请求预算，Reader 另受 16 次上限约束；每分片最多 6 个原生动作，整轮最多 460 个原生动作。可通过 CLI 修改这些通用预算。请求失败也计入次数，没有额外隐藏重试。新入口的预算是整轮预算，不是旧入口的每阶段预算。

连续错误次数默认不再终止任务（`--max-consecutive-errors 0`）；错误会记录后继续请求，总请求数、动作数、墙钟时间、仿真时间和跌倒终止仍有效。设为正整数可恢复旧的连续错误终止策略。协议错误日志同时记录安全错误说明、响应结束类型和已解析的工具名，便于区分不可用工具与响应格式错误；这些诊断只写入日志，不改变模型反馈或工具校验。

每轮保存 `metrics.json`、`events.jsonl`、模型请求原图、`evaluation.evaluator-only.json`、物理轨迹、原始模型/位姿录像数据；有记忆组另有 `memory.json` 和 `memory.md`。新导出器根据实际静态场景和轨迹构图，不要求存在 A/B 对象；导出后完整解码验证帧数，不需要再次请求模型：

```bash
MUJOCO_GL=egl python -m go2_demo.general_video runs/general-occluded-return-new/capture
```

当前新增版本应先经过离线协议、状态推进、缓存、信息边界和实际物理接口检查。模型视觉定位质量、A/B 成功率与记忆收益需要另行开展新的付费对照，不能沿用旧版本分数。

2026-11-25 恢复默认程序判断模式后，通用子任务、Agent、运行、记忆和执行模块的 **213 项离线回归测试通过**。此前两种场景的真实 MuJoCo 接口检查也均通过：短距离运动、当前像素绑定、表面点接近与停稳、按需记忆、日志和通用录像导出；导出视频均完整解码验证。物理检查通过本地 MockTransport 提供脚本响应，**外部 API 请求为 0**。脚本刻意不完成目标任务，独立任务评分保持失败，验证局部技能成功不会被错误升级为任务成功。检查数据位于 `runs/general-agent-offline-smoke-r2/`，其中的请求计数是本地模拟调用，不是付费调用。

可以运行实际物理接口的离线检查，HTTP 全部由本地 MockTransport 拦截，不读取密钥或发起联网请求。当前脚本显式覆盖可选的 `agent` 更新模式；默认 `measured` 的状态推进和独立评分由上面的回归测试覆盖。脚本在两种场景中做短距离运动、像素几何绑定、记忆读取与提前结束，再导出录像并验证完整解码；它不执行完整任务，也不评价大模型识别能力：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python scripts/check_general_agent_offline.py --output runs/general-offline-check-new
```
