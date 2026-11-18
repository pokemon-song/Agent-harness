# 到 B 后原路返回

让一台 Go2 从起点绕过抽屉柜，穿过绿色门框，到达字母 B 标牌。随后清空规划器的短期上下文，看它能否借助自己在去程留下的路线记忆，从 B 沿原路走回真实起点。仿真场景名为 `composite_drawers_gate_b`。 [场景定义](../go2_demo/scenarios.py) · [往返任务协议](../go2_demo/roundtrip_mission.py)

## 给测试同学的任务说明

1. **去程。** 机器人自行选择从柜体哪一侧绕过，再从绿门开口内通过。以机身中心为准，过门后沿穿门方向至少前进 0.8 米。在门后的 A、B、C 三块白色字母标牌中找到 B，面向 B 停在距其中心 0.82–1.18 米处。停稳后调用第一次 `finish`，结束去程。
2. **记忆交接。** 去程记忆 Agent 根据公开的相机观测、几何测量、里程计和实际动作反馈写 Markdown，记录实际绕行侧、地标、转折位置和测量误差。去程通过独立验收、记忆写入完成后，清空规划器的对话历史、临时笔记和跟踪点。机器人留在 B 前，姿态、场景、仿真时间和里程计原点连续。去程失败或没有可读记忆时，返程记为未运行。
3. **返程。** 新的规划上下文须先读取去程实际保存的只读记忆，再结合当前观测自行选动作。应反向穿过同一绿门，从柜体去程经过的那一侧绕回起点。绕行侧以柜体在场地中的位置为准，掉头后机器人的左右方向会改变。回到实际起点 0.30 米范围内，停稳后调用第二次 `finish`。终点朝向不限。

评分器使用机器人的实测轨迹，不向规划器提供标准路线。沿轨迹每 0.05 米采样，返程点到去程路线的最大偏差须不超过 0.50 米，至少 95% 的去程路线须被返程覆盖。停稳要求线速度低于 0.12 米/秒、绝对角速度低于 0.25 弧度/秒。历史版本要求全程无碰撞、无跌倒，两段均通过才算往返成功。 [独立轨迹评分](../go2_demo/roundtrip.py) · [完整任务与预算](md-roundtrip-task.md)

## 在新环境中复现

在 Linux 或 WSL2 中检出发布分支 `repro/roundtrip-memory-20261118`，进入仓库根目录。以下命令使用 Python 3.11、CPU 版 PyTorch 和 EGL 离屏渲染。第一次安装和资产下载需要联网。 [Python 依赖](../pyproject.toml) · [资产清单](../assets.lock.json) · [资产下载与校验脚本](../scripts/fetch_assets.py)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[video]'
python scripts/fetch_assets.py
python scripts/fetch_assets.py --verify
```

将自己的 DeepSeek API 密钥放在仓库外的文本文件中，将下方示例路径替换为该文件的路径。运行会使用付费 API。去程规划最多 60 个请求、去程记忆最多 48 个请求、返程规划最多 90 个请求。每段墙钟和仿真时间上限分别为 1,800 秒和 300 秒。 [实验入口](../scripts/run_md_roundtrip.py)

```bash
MUJOCO_GL=egl python scripts/run_md_roundtrip.py --api-key-file /path/to/your-key.txt
```

完成后，使用终端打印的本轮目录生成报告和连续视频。每次运行保存独立记录。可在汇总中查看两段是否通过、返程最大路线偏差、去程路线覆盖率、终点误差、碰撞和跌倒情况。 [报告与录像生成脚本](../scripts/report_md_roundtrip.py)

```bash
read -r -p '粘贴本轮运行目录：' RUN_DIRECTORY
MUJOCO_GL=egl python scripts/report_md_roundtrip.py "$RUN_DIRECTORY" --render
```

## 供对照的真实运行

2026 年 11 月 11 日，使用源码提交 `987fbb3fcf55eb3098935cd245031faa5f2dcdf2` 和 `deepseek-flash` 的一次实测中，机器人通过了去程和返程。返程覆盖了全部去程路线，最大偏差 **0.167 米**；终点距起点 **0.171 米**，全程无碰撞、无跌倒。这是单轮结果，新一轮模型动作可能不同。 [实测报告](../results/md-roundtrip-20261111/REPORT.zh-CN.md) · [机器可读结果](../results/md-roundtrip-20261111/summary.json) · [冻结配置](../results/md-roundtrip-20261111/experiment.json)

119 秒录像按仿真时间展示两段连续运动。查看记忆内容与动作的关系，可对照最终发布的记忆索引和独立审计。 [完整 MP4](../results/md-roundtrip-20261111/third_person/third-person-sim.mp4) · [实际记忆索引](../results/md-roundtrip-20261111/memory/revisions/00000009-44493da2390442e398900d2422161c72/MEMORY.md) · [独立审计](../results/md-roundtrip-20261111/independent-audit.json)
