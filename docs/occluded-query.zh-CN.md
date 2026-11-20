# A/B 遮挡返回：直接阅读与按需问答

本版本比较两种返程记忆接口。两组复用 seed 0 历史 H2 从出发到首次在蓝球 A 前合格停稳的真实去程。程序在 MuJoCo 中重新执行去程物理动作，逐张核对 121 张历史规划输入和 6,457 个控制步的测量，再交给全新规划器寻找绿方块 B。返程动作由对应组的模型实时决定。去程回放不调用模型。

记忆在 A 处由同一套规则根据已发生的公开 RGB-D 观测和里程计生成，一次性封存 6 篇 MD，包括两种目标的历史观测、分段路线和最后一次实测姿态。证据条目带帧号、时间和来源。规划器只能通过本组的记忆工具获取正文，不能访问私有评分和轨迹。两组保持共同里程计，清空去程规划历史、工具缓存和跟踪点，使用相同的当前 RGB、当前颜色候选和运动接口。

| 接口 | 规划器能获得什么 |
| --- | --- |
| `memory_read(path)` | 指定的一篇完整文档。 |
| `memory_query(question)` | 独立只读记忆模型从相同的文档中选取最多 3 条原始证据，按问题相关性排序，附来源及支持程度。没有证据时可返回空列表。 |

成功取回的内容保留在当前观测中，下一次成功读取或提问时替换。工具回执和过去六轮的观测不重复携带该字段。两组首次运动或 finish 前都需要一次成功读取。相同快照中重复相同问题可使用缓存。问答模型没有机器人控制工具。

规划及问答均使用官方 DeepSeek 的 `deepseek-flash`。每组返程最多 115 次新请求，所有规划、问答和重试均计入同一额度，单请求超时 90 秒、最大输出 8,192 tokens。返程墙钟上限 1,117 秒、仿真上限 470 秒。首个与场景物体接触的物理步结束整轮。B 须满足距离 0.82–1.18 米、朝向误差 ≤0.25 rad、停稳及 finish，由原 A/B 评分器验收。A 的历史合格见证和严格安全口径下的最终整轮得分分别记录。

在装有原项目依赖、已按 `assets.lock.json` 准备好 `assets/` 的工作树中运行。将 `MODEL_API_KEY_FILE` 指向仓库外的密钥文件。两个在线命令都会向官方端点发送公开仿真数据并产生 API 费用。

```bash
unzip -q datasets/occluded-return-seed0.zip -d runs/outbound-seed0-input
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
common=(--archive runs/outbound-seed0-input --runs-dir runs/paired-return)
python -m go2_demo occluded-return --mode prepare "${common[@]}"
python -m go2_demo occluded-return --mode direct "${common[@]}" --api-key-file "$MODEL_API_KEY_FILE"
python -m go2_demo occluded-return --mode query "${common[@]}" --api-key-file "$MODEL_API_KEY_FILE"
python -m go2_demo occluded-return --mode render --runs-dir runs/paired-return
```

`shared/documents/` 保存实际的交接记忆。`direct/` 和 `query/` 各自保存评分、分角色用量、规划事件、前视图像和完整连续轨迹。录像 `video/full.mp4` 包含共同的去程以及该组独立的返程，按仿真时间播放。每组的 `manifest.json` 留存代码、A 处观测、去程轨迹和文档的哈希值。

## 根据当前位置询问返程路段

`local-query` 把规划器提问时的公开里程计和任务阶段传给记忆模型。在去程路点中找出离当前位置最近的记录及按返程顺序相邻的两个记录，作为选取证据的提示。记忆模型从同一套冻结文档中选取至多三条记录，并指出证据缺口。记录过机身位置的路点仍须结合当前观测检查通行条件。缓存按问题、位姿、阶段和本轮文档隔离。

在已完成 `prepare` 的运行目录执行新版完整返程并生成录像：

```bash
python -m go2_demo occluded-return --mode local-query "${common[@]}" --api-key-file "$MODEL_API_KEY_FILE"
python -m go2_demo occluded-return --mode render --runs-dir runs/paired-return
```

可单独复测原问答组第 6 次决策。在同一张图、同一物理状态和同一份五轮公开历史下，仅替换当前回答。旧进程未归档的模型内部历史无法还原，所以对照组也重新请求模型。两组使用相同的原问题、工具和 8,192 输出 token 预算；每个样本只提请规划器做一次决策，物理工具在各自恢复的同一状态上执行。再次查询记忆单列为追问。每个返回工具的样本留存执行结果和执行后的图片。以下命令有一次记忆选择和十次规划请求。

```bash
python scripts/retest_occluded_frame.py --archive runs/outbound-seed0-input --assets-dir assets \
    --out runs/frame6-local-comparison --api-key-file "$MODEL_API_KEY_FILE" --pairs 5
```
