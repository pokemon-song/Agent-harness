# A→B 遮挡导航复现说明

机器人从起点观察绿色方块 B，绕过抽屉柜并在蓝球 A 前停稳，然后找回 B、在 B 前停稳并调用 `finish`。A、B 停靠均要求机身距目标中心 0.82–1.18 米、朝向误差不超过 0.25 弧度。评分器独立记录 A 的途中停靠和 B 的最终停靠。场景为 `memory_ab_occluded_v1`。[场景与指令](../go2_demo/scenarios.py) · [检查点评分器](../go2_demo/ordered_evaluation.py)

H0 接收当前 RGB、公开观测和最近六轮对话。H2 的规划器接收当前结构化观测和规则记忆，两段行程在同一轮连续执行。本实验的返回目标是绿色方块 B。[规划输入](../go2_demo/models.py) · [H2 记忆与规划](../go2_demo/memory_planning.py)

## 历史录像与当前评分

| | 2026-11-18 已完成的 H0/H2 实测 | 本分支的当前实现 |
| --- | --- | --- |
| 碰撞后 | 匹配提交 `4633bc3708c5669667c6fecd05da8f1446759c2c` 的严格规则。柜体接触可终止任务并撤销途中 A 的最终得分。 | 碰撞记入日志，机器人可继续执行动作。持续接触同一物体只扣一次，柜身与把手合并统计。 |
| 评分 | 独立评分器按两段停靠、全程安全和 `finish` 判定整轮通过。 | 合格停靠 A 得 50 分，完成 B 并正常结束再得 50 分，每次独立碰撞扣 5 分，最低 0 分。碰撞扣分与整轮通过分别报告，跌倒仍结束任务。 |
| 模型实测 | 同一 `seed=0`，使用官方 `deepseek-flash`，逐组记录如下。 | 新评分规则尚未启动模型复测。 |

| 组别 | 决策 / 已确认请求 | 旧规则结果 | 完整录像和原始记录 |
| --- | ---: | --- | --- |
| H0 无外部记忆 | 47 / 47 | 第 47 步碰柜终止，A、B 均未通过。 | [录像 · 2 分 46 秒](../results/20261118-extended240/H0/H0-full-replay.mp4) · [评分](../results/20261118-extended240/H0/metrics.json) · [模型和预算](../results/20261118-extended240/H0/manifest.json) |
| H2 外部记忆 | 155 / 158 | 第 121 步首次合格停在 A 前，距 A 1.162 米。第 155 步返程碰撞，终点距 B 3.50 米，A、B 最终均未得分。 | [录像 · 4 分 52 秒](../results/20261118-extended240/H2/H2-full-replay.mp4) · [评分](../results/20261118-extended240/H2/metrics.json) · [模型和预算](../results/20261118-extended240/H2/manifest.json) |

H2 在前 99 次决策完成后遇到一次进程中断。恢复前核对了机器人状态、100 份模型输入和完整记忆，续跑从同一位置继续。计入中断时未获回执的请求，实际请求总数上限为 160。视频按照原始物理控制记录重放并压缩模型等待时间。[H2 恢复计数](../results/20261118-extended240/H2/metrics.json)

## 准备环境

从本分支根目录运行以下命令即可使用当前评分。要复现录像对应的旧规则，可先建立独立工作树，并将本次使用的官方 DeepSeek 兼容入口复制过去。两种版本使用相同命令和资产。

```bash
# 仅在要运行旧规则时执行
git worktree add --detach ../occluded-strict 4633bc3708c5669667c6fecd05da8f1446759c2c
cp scripts/run_deepseek_memory.py ../occluded-strict/scripts/
cd ../occluded-strict
```

历史环境使用 Linux、Python 3.11.15、MuJoCo 3.2.5 和 PyTorch 2.5.1+cpu。安装依赖后，下载并校验 21 个固定资产。[项目依赖](../pyproject.toml) · [资产清单](../assets.lock.json) · [下载及校验程序](../scripts/fetch_assets.py) · [第三方许可](../THIRD_PARTY_NOTICES.md)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 opencv-python==4.11.0.86 httpx==0.28.1 Pillow==11.3.0
python -m pip install -e .
python scripts/fetch_assets.py
python scripts/fetch_assets.py --verify
```

## 运行对照

在同一 Linux Bash 会话内，从所选工作树根目录运行。先把自己的 DeepSeek API 密钥保存在仓库外的私有文件，再把下面的路径指向该文件。两条运行命令会向官方 API 发送仿真观测并按账户资费计费。每组总请求上限为 240，H2 的协议重试计入该额度。真实模型的决策序列可能随服务端和采样变化。[运行参数](../go2_demo/__main__.py) · [H2 官方兼容入口](../scripts/run_deepseek_memory.py)

```bash
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MODEL_API_KEY_FILE="$HOME/.config/go2/deepseek.key"
common=(--task memory_ab_occluded_v1 --provider deepseek --model-name deepseek-flash
        --api-key-file "$MODEL_API_KEY_FILE"
        --seed 0 --max-model-calls 240 --max-wall-seconds 2400
        --max-sim-seconds 600 --api-timeout 90 --max-output-tokens 8192
        --history-turns 6 --no-env-proxy)

python -m go2_demo "${common[@]}" --runs-dir runs/reproduction/H0
python scripts/run_deepseek_memory.py "${common[@]}" --runs-dir runs/reproduction/H2
```

H2 入口启用 `--memory-v1`，针对官方端点的思考请求把 `tool_choice="required"` 调整为 `auto`，规划约束和动作校验继续由同一套执行代码负责。两组的新运行目录都会写出 `manifest.json`、`metrics.json`、动作事件、仿真轨迹和决策截图。H2 还会写出本轮的记忆文档。新规则下可读取 `metrics.json` 的 `success`、`goal_points`、`collision_count`、`collision_penalty` 和 `score`。[结果记录与计分](../go2_demo/runner.py) · [碰撞事件采集](../go2_demo/backend.py)
