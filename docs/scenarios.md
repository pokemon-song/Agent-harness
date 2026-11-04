# 可运行的新测试场景

QUARD 离线样本读取和 MuJoCo 闭环测试有各自的入口。现有13个导航场景之外，新增本地自建复合任务 `composite_drawers_gate_b`。DeepSeek 启动脚本没有指定任务时默认选择蓝色球体，原始红色标记保留为 `red_marker`。

当前机器人运行方式仅保留开启思考的 API。DeepSeek 从 `../deepseekapi.txt` 读取凭据，并在多轮工具调用中回传完整思考字段。

在项目目录直接运行：

```bash
# 查看全部任务，不加载密钥或启动仿真
./scripts/run_deepseek.sh --list-tasks

# 三块白色牌中选择字母 B
./scripts/run_deepseek.sh --task distinguish_texture_b --viewer

# 绕过沙发，接近后方目标
./scripts/run_deepseek.sh --task go_avoid_texture_sofa --viewer

# 穿过三角形门框
./scripts/run_deepseek.sh --task go_through_red_triangle_tunnel --viewer

# 绕抽屉柜→穿绿色矩形门→从A/B/C中接近B并停稳
./scripts/run_deepseek.sh --task composite_drawers_gate_b --viewer --seed 0

# 旧范围：13项导航，另列2项不支持的爬行跳过
./scripts/run_deepseek.sh --suite quard-planar --viewer

# 只运行1项复合任务
./scripts/run_deepseek.sh --suite composite --viewer

# 扩展范围：14项可运行任务，另列2项爬行跳过
./scripts/run_deepseek.sh --suite extended-planar --viewer
```

`--list-tasks` 只列任务；其余运行命令会调用模型API。不开窗口时去掉 `--viewer`。每项任务的预算沿用 `--max-model-calls`、`--max-wall-seconds` 和 `--max-sim-seconds`；批量入口为每项任务重置模型历史和仿真状态，汇总 API 调用与 token 用量。默认 API 超时 90 秒、每次输出预算 8192 token（包含思考）、每项墙钟预算 300 秒；可用 `--api-timeout`、`--max-output-tokens` 和 `--max-wall-seconds` 调整。

| 类型 | 任务 ID | 窗口中的实际内容 |
| --- | --- | --- |
| 目标到达 | `go_to_blue_ball`、`go_to_green_cube`、`go_to_red_cube` | 对应颜色和形状的物体，另有异色干扰物 |
| 字母辨别 | `distinguish_texture_a`、`distinguish_texture_b`、`distinguish_texture_c` | 同时出现 A/B/C 三块同色标牌，需要选择指令指定字母 |
| 绕障 | `go_avoid_texture_cooker`、`go_avoid_texture_drawers`、`go_avoid_texture_fan`、`go_avoid_texture_sofa` | 炉台、抽屉柜、风扇或沙发挡在目标前，接触障碍会记为失败 |
| 门框穿越 | `go_through_green_rectangle_tunnel`、`go_through_red_triangle_tunnel`、`go_through_yellow_triangle_tunnel` | 绿色矩形、红色三角形或黄色三角形门框，出口后有参照标记 |
| 复合任务 `composite` | `composite_drawers_gate_b` | 先绕过抽屉柜，再穿过绿色矩形门，最后从A/B/C标牌中接近B并停稳 |
| 低姿态爬行 | `crawl_red_gate2`、`crawl_red_gate3` | 当前不支持；列出原因并跳过，不启动模型或计为成功 |

原13项导航采用QUARD公开任务名称与类型，在当前Go2/MuJoCo中重新构造几何。家具是基础几何体组合，门框允许现有平地步态直立穿行；它们属于本项目的场景改编，不是论文原始Isaac Gym环境、纹理、窄隧道净空或Uv/Ul划分。新增复合任务是本地自建场景，不是QUARD官方任务。真实QUARD四帧数据的检查方式仍见 [离线数据测试](quard.md)。

机器人已新增 `set_posture(body_height_m, body_pitch_rad)` 静止高度／仰角调整和 `reset_posture()` 恢复行走模式。静止调姿不会使当前步态具备低姿态行走能力：调姿模式中移动／转向会返回 `POSTURE_MODE_ACTIVE`，停止保持姿态，须显式恢复后才能继续行走。两项爬行任务仍跳过，接口语义见 [运行与接口说明](demo.md)。

## 运行结果与评分

每项任务在 `runs/` 中保存自己的 `manifest.json`、`events.jsonl`、`trajectory.jsonl`、`metrics.json` 和相机图像；任务 ID 和实际指令会打印在终端，并写入清单。批量结果额外保存在 `runs/<批次>/summary.json`。

目标和字母任务检查距正确目标中心 `1.0±0.18 m`、朝向误差不超过 `0.25 rad`、停止且没有碰撞或跌倒。选错字母或物体不会因为调用 `finish()` 获得成功。

穿门任务检查实际轨迹是否从开口内部向前越过门框，并到达门后至少 0.8 米的停止区域；只站在门前或从门外绕行不能通过。该完成余量现已明确写入模型指令，并提供相机标定和实际机身里程计帮助判断进度，见 [修复说明](passage-solution.md)。所有障碍和干扰物都参与真实物理碰撞。任务目标真值只进入独立评分，不作为坐标直接提供给模型。

`summary.json` 分别统计成功、失败、跳过和未运行项。两项爬行任务的 `success=null`，不进入可运行任务的成功数；某项出错后其他任务仍可继续，主动关闭窗口或中断则停止批次并保留未运行项。

单元测试和 QUARD 离线数据验证继续保留，用于检查接口与数据格式；机器人决策统一调用模型 API。运行日志只记录思考字符数，原始思考文本仅供进程内历史回传。

## 复合任务与评测版本

`composite_drawers_gate_b` 在真实MuJoCo关节动力学中串联三段目标。独立评分从实际base轨迹确认先到抽屉柜后方，再从绿色门开口向前跨越并到门后至少0.8米；最终在正确B标牌中心的水平距离 `1.0±0.18 m` 内、朝向误差不超过 `0.25 rad`，水平速度小于 `0.12 m/s`、偏航角速度绝对值小于 `0.25 rad/s`。还须全程无碰撞／跌倒、未超预算并正常调用 `finish`。

`stages` 和 `stage_completion_rate` 只用于诊断进度，不把“过柜”“过门”单独当作严格成功或部分主分。只在B附近结束、从门外绕行、顺序不符合轨迹验收或提前 `finish`，都不能通过。验收使用base参考轨迹与物理碰撞记录，未单独验证完整机器人包络或扫掠净空。

默认模型评测v2仍是原13任务×5次、四类各25%。显式v3增加复合类后为14任务×5次、五类各20%，沿用同一五维分项和预算；两版不混排，也不改变既有v2的82.96分。场景清单见 [datasets/composite/dataset.json](../datasets/composite/dataset.json)，实际仿真由 [scenarios.py](../go2_demo/scenarios.py) 加载。详细命令、阶段验收和验证边界见 [复合测试说明](composite-tests.md)，正式评分见 [模型评测协议](model-evaluation.md)。

本次新增任务验证不调用付费模型API。单测、合成轨迹检查和预定动作的MuJoCo可达性验证分别标明来源，不能当作模型成功率；尚无复合任务或v3模型成绩。固定seed=0的5次重复也不是5个随机新场景。

已保存 [实际MuJoCo可达性通过记录](../runs/20261104-024854-composite-scripted-validation-3271f2/validation.json)：预定动作完成三阶段并停稳，无碰撞／跌倒，`api_requests=0`、`is_llm_result=false`、`model_score=null`。首次因末段朝向超限而未通过的记录也完整保留，见 [复合测试说明](composite-tests.md)；这些结果不计入模型成绩。

## 历史实测（2026-10-28，规则策略／非思考 API）

以下是切换到思考 API 前保存的历史结果，不是当前模式的成绩。两个批次均运行种子 0、每项最多 25 次决策。规则策略使用每项 90 秒墙钟预算，非思考 DeepSeek 使用每项 180 秒；两者均保留原来的物理控制器、动作限制和停止确认。

| 类别 | 场景数 | 规则策略通过 | DeepSeek 通过 |
| --- | --- | --- | --- |
| 目标到达 | 3 | 3 | 0 |
| 字母辨别 | 3 | 3 | 0 |
| 绕障 | 4 | 0 | 0 |
| 门框穿越 | 3 | 3 | 1 |
| 合计 | 13 | 9 | 1 |

两个批次分别额外记录 2 项不支持的爬行任务，没有未运行项。原始结果：[规则策略汇总](../runs/scenario-scripted-validation/20261028-182841-suite-quard-planar-scripted-s0-550ada/summary.json)、[DeepSeek 汇总](../runs/scenario-api-validation/20261028-182841-suite-quard-planar-api-s0-828b57/summary.json)。

DeepSeek 成功通过红色三角形门框；实测机身已从开口内穿过并停在出口区域，没有碰撞或跌倒。其他场景主要因反复选点、未声明完成而耗尽预算，另有一次绕障碰撞，以及蓝球任务停在 1.195 m、超出距离验收范围。部分任务几何已合格但没有调用 `finish`，仍判为失败。原始日志未修改，不将原始红色标记的成功复用到新场景。

字母 B 曾通过一次真实 WSLg 窗口与规则策略闭环验证，记录见 [viewer metrics](../runs/scenario-viewer-validation/20261028-182939-scripted-distinguish_texture_b-s0-a87196/metrics.json)。这些历史结果只对应当时的本地改编场景及预算，不代表当前思考 API 模式或论文 QUARD 原始评测成绩。当前验证记录见 [validation.md](validation.md)。
