# 本地复合测试：绕柜、穿门与识别B

新增任务 `composite_drawers_gate_b` 把绕障、穿门、字母辨别和最终停稳放进同一个真实MuJoCo回合，类别为 `composite`。机器人需要先绕过抽屉柜，再穿过绿色矩形门，最后从A/B/C三块标牌中接近B。模型仍使用开启思考的视觉与工具API，自行选点和决定动作；Harness沿用几何跟踪、短程动作与独立评分，不替模型规划路线。

这是本项目自建的复合场景，不是QUARD官方任务。可读说明清单为 [datasets/composite/dataset.json](../datasets/composite/dataset.json)，包含公开指令、几何布局、阶段／验收配置与来源，由Scenario导出。实际仿真配置由 [go2_demo/scenarios.py](../go2_demo/scenarios.py) 加载；模型不会直接读取清单中的目标真值。

## 运行入口

先激活项目环境。列任务不会加载密钥或启动仿真：

```bash
source .venv/bin/activate
python -m go2_demo --list-tasks
```

以下单项或套件命令会调用模型API并产生费用，供之后实际评测时使用；本次新增任务的实现验证没有执行这些付费模型回合：

```bash
# 单项任务，使用已有DeepSeek凭据配置
python -m go2_demo --provider deepseek \
  --api-key-file ../deepseekapi.txt --no-env-proxy \
  --task composite_drawers_gate_b --viewer --seed 0

# 通过现有启动脚本运行同一任务
./scripts/run_deepseek.sh --task composite_drawers_gate_b --viewer --seed 0

# composite只包含新增1项；extended-planar包含原13项＋复合1项
./scripts/run_deepseek.sh --suite composite --viewer --seed 0
./scripts/run_deepseek.sh --suite extended-planar --viewer --seed 0
```

不需要窗口时去掉 `--viewer`。旧 `--suite quard-planar` 保持原13项导航，另列2项不支持的爬行跳过；`extended-planar` 为14项可运行任务，另列同样2项跳过。两项低姿态爬行没有因新增复合任务而变为可运行。

## 独立验收与顺序

| 阶段 | 独立验收条件 |
| --- | --- |
| `avoid_drawers` | base原点向前越过抽屉柜后的验收平面 `x=2.3 m` |
| `pass_gate` | 在前一阶段之后，base向前跨越 `x=3.4 m` 的门平面时 `y∈[-0.60,0.60] m`，继续到 `x=4.2 m`，即门后至少0.8米 |
| `approach_b` | 最终距正确B中心水平距离 `1.0±0.18 m`，朝向误差≤`0.25 rad`，速度<`0.12 m/s`且偏航角速度绝对值<`0.25 rad/s` |

轨迹事件必须形成“过柜后→从开口穿门→到门后”的正确顺序链，最终位置也须在 `x≥4.2 m`。允许回退和重新尝试，不要求整条路径的x坐标单调增加；尚未达到门后0.8米余量就退回门前，会撤销未完成的过门事件，须从开口重新向前穿过。碰撞、跌倒及预算仍按整个回合累计。验收以base参考点与真实物理碰撞记录为依据，没有单独计算完整机器人包络或整机扫掠净空。

`metrics.json` 的 `evaluation` 中，`stages={avoid_drawers,pass_gate,approach_b}` 和 `stage_completion_rate=sum(stages)/3` 只说明已满足哪些阶段条件。部分阶段达成不构成严格成功，也不给部分主分。全部要求满足、全程无碰撞／跌倒、未超预算、正常 `finish` 且最终停稳，才是成功回合。模型声明完成不会替代独立轨迹和目标验收。

阶段率为1也不保证整回合成功：例如已有三个阶段的几何证据，但累计碰撞或跌倒，仍会被全局安全条件拒绝。

因此，直接绕到B附近但没有通过门开口、只完成过柜／过门后提前结束，或在门外绕过门框，都不能通过整项任务。所有阶段共用一个回合预算，不为每段各增加30次调用或300秒。

## 显式v3评测计划

默认v2保留原13任务×5次＝65回合，四类各25%。新增类别只在显式选择 `go2-model-eval-v3` 时进入标准模型评测：14任务×5次＝70回合，五类各20%。

| 类别 | 任务数 | 每任务重复 | 回合数 | v3权重 |
| --- | ---: | ---: | ---: | ---: |
| `go_to` | 3 | 5 | 15 | 20% |
| `distinguish` | 3 | 5 | 15 | 20% |
| `go_avoid` | 4 | 5 | 20 | 20% |
| `go_through` | 3 | 5 | 15 | 20% |
| `composite` | 1 | 5 | 5 | 20% |

两版每回合均使用 `S×(60+20P+10W+5T+5C)`：严格成功60分、路径20分、墙钟10分、仿真5分、调用5分；有效失败所有分项为0。固定预算仍为30次模型调用、300秒墙钟、120秒仿真、6轮近期完整工具历史、8192输出token（含思考）、单次API超时90秒。

新建v3计划只冻结配置，不联网、不读取密钥：

```bash
python -m go2_demo.benchmark_cli plan \
  --protocol go2-model-eval-v3 \
  --label deepseek-composite-v3 --provider deepseek --model-name deepseek-flash \
  --output-dir runs/model-eval/deepseek-composite-v3 --repeats 5
```

准备实际付费评测后才运行下面的命令；它未在本次实现验证中执行：

```bash
python -m go2_demo.benchmark_cli run \
  --plan runs/model-eval/deepseek-composite-v3/plan.json \
  --api-key-file ../deepseekapi.txt --no-env-proxy

# 已有结果的离线评分，不调用模型
python -m go2_demo.benchmark_cli report \
  --plan runs/model-eval/deepseek-composite-v3/plan.json \
  --output runs/model-eval/deepseek-composite-v3/report.json
```

综合分先对同任务5次重复平均，再对组内任务平均，最后五类等权。仅1项任务的复合类仍占20%，因此总分不等同于70回合微平均。路径效率使用起点到最终B停靠区域的直线距离下界，忽略绕柜和穿门的中间约束；它不是最短可行路径或SPL，同任务对所有模型使用同一参照。完整定义见 [模型评测协议](model-evaluation.md)。

## 不调用模型的验证与已有结果

实现回归可运行：

```bash
python -m pytest -q \
  tests/test_scenarios.py tests/test_scenario_backend.py \
  tests/test_scenario_cli.py tests/test_scenario_runner.py \
  tests/test_benchmark.py tests/test_benchmark_cli.py
```

这些检查验证场景、合成轨迹的顺序判定、物理后端和离线评分接口，不产生模型成绩。真实MuJoCo可达性若使用预定动作序列，须明确记录为 `scripted validation`；它只能证明该控制序列在该场景下的可达性，不能当作模型成功、模型成功率或v3模型成绩。测试通过数以实际输出为准。

2026-11-04本次完整 `pytest -q` 结果为 **492 passed in 10.60s**。这些单元／回归检查不调用真实模型API，不计为模型评测回合。

实际MuJoCo可达性检查入口为：

```bash
.venv/bin/python scripts/validate_composite.py
```

该脚本通过真实MuJoCo与Harness执行预定waypoints，使用仿真位姿，明确记录 `is_llm_result=false`、`api_requests=0`、`model_score=null`、`uses_privileged_simulator_pose=true`。这是一项scripted validation，不创建模型成功样本；它也不使用模型API或读取模型密钥。

2026-11-04已保存两次实际MuJoCo可达性检查。初次脚本没有补齐最后对准B的动作，朝向误差 `0.256898 rad > 0.25 rad`，被原阈值正确拒绝；补齐末段转向后，三个阶段全部通过、距B中心约 `1.062521 m`、朝向误差约 `0.063864 rad`，无碰撞、无跌倒。两次原始证据都保留，验收阈值没有放宽：

| 预定动作验证 | 独立验收 | 原始记录 |
| --- | --- | --- |
| 初次缺少最终对准 | 2/3阶段，最终朝向超限，未通过 | [validation.json](../runs/20261104-024823-composite-scripted-validation-276989/validation.json) |
| 补齐最终转向 | 3/3阶段，最终停稳及安全条件通过 | [validation.json](../runs/20261104-024854-composite-scripted-validation-3271f2/validation.json)、[初始原图](../runs/20261104-024854-composite-scripted-validation-3271f2/initial.png)、[终态原图](../runs/20261104-024854-composite-scripted-validation-3271f2/final.png) |

这里的3/3是预定动作的阶段验收结果，不是模型成功率或正式模型得分。

既有 [v2阶段报告](../runs/model-eval/20261103-deepseek-flash-vs-qwen-max/REPORT.zh-CN.md) 中，DeepSeek在原65回合严格成功60次，综合分82.96。Qwen保留因账号服务阻塞而中止的阶段记录，未形成完整跨模型排名。新增复合任务不会改写旧计划、原始日志、82.96分或其报告；v2与v3不混排，也不能把旧回合补作复合任务成绩。本次新增内容尚未调用付费模型API，尚无复合任务或v3模型成绩。

场景seed固定为0，当前目标抖动为0。5次重复主要观察模型／服务输出波动，不能称为5个随机场景或据此宣称泛化性能。
