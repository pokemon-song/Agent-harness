# 单班导航记忆：去 B 后原路返回

去程到 B 并原路返回通过。

返程终点距起点 **0.171 米**，相对去程折线最大／平均偏差 **0.167／0.087 米**，在 0.50 米走廊内的去程采样覆盖 **100%**；往返均从柜体同侧通过，绿门穿越方向相反，全程无碰撞和跌倒。整趟实际墙钟 **20.6 分钟**，模型等待期间仿真暂停。

去程记忆发布到 v9，39 条事件全部处理；返程规划器实际读取 MD 13 次，首次转身前已经读取路线及测量片段。50 项独立记录／完整性核查通过，详见 [独立审计](independent-audit.json) 和 [记忆内容与限制](findings.md)。自由文本仍有少量数值解释和证据粒度问题，不能把任务成功等同于全部语义正确。

本轮共享坐标字段使普通反馈保守回退为完整输入，没有字段压缩收益。运行结束后已在 `f937bab` 补兼容并通过 1,174 项回归；[同数据离线投影核验](postprocess/projection-compatibility.json) 不计入本次模型成绩，没有重新调用模型。下面的实际成绩始终属于冻结提交 `987fbb3`。

真实模型结果：True；模型：`deepseek-flash`；冻结提交：`987fbb3fcf55eb3098935cd245031faa5f2dcdf2`。

本实验设计为同一个机器人世界中的连续往返：交接时清空规划模型上下文，保留机器人状态和同一个起点坐标系。返程从实际发布的去程 MD 检索路线，不由程序填充返程答案。实际连续性和只读检查见下方 checks；模型读取文件不等于已证明记忆的因果收益。

[播放页面](watch.html) · [原始汇总](summary.json) · [冻结配置](experiment.json) · [结构化分析](analysis.json)

声明预算：`{"budgets": {"outbound": {"max_model_calls": 60, "max_wall_seconds": 1800.0, "max_sim_seconds": 300.0}, "return": {"max_model_calls": 90, "max_wall_seconds": 1800.0, "max_sim_seconds": 300.0}}, "memory_calls": 48, "api_timeout_s": 180.0, "max_output_tokens": 16384, "history_turns": 2}`。

| 阶段 | 结果 | 终止/错误 | 墙钟秒 | 仿真秒 | 路径米 | 规划/记忆请求 |
|---|---|---|---:|---:|---:|---:|
| 去程：绕柜、过门、到 B | 通过 | `finish` | 928.11 | 56.88 | 6.730 | 59/41 |
| 返程：读取去程记忆并原路返回 | 通过 | `finish` | 304.74 | 62.08 | 6.504 | 38/0 |

请求合计：138，完整回复：138；已知用量：`{"prompt_tokens": 2881052, "completion_tokens": 326862, "total_tokens": 3207914}`；用量未知尝试：0；用量完整：True。失败请求缺失的用量不填零，不按输出上限推算费用。

[完整连续第三视角录像](third_person/third-person-sim.mp4)：119.00秒，2975帧。按仿真时间播放，推理暂停从时轴中移除；没有 RESET 卡，不重新执行物理仿真。

## 去程：绕柜、过门、到 B

- 规划 Agent：请求59，完整回复59，未知用量尝试0；已知用量 `{"prompt_tokens": 1034483, "completion_tokens": 126566, "total_tokens": 1161049}`。
  请求错误：`[]`；worker 生命周期错误：`[]`。
- 记忆 Agent：请求41，完整回复41，未知用量尝试0；已知用量 `{"prompt_tokens": 1144047, "completion_tokens": 153558, "total_tokens": 1297605}`。
  请求错误：`[]`；worker 生命周期错误：`[]`。

模型等待：775.49秒；记忆等待：147.35秒；工具执行墙钟：3.61秒。

实际成功读取 MD 9次；首个成功运动前已有读取：True。这核对读工具记录，不自动证明路线语义正确或规划确实采用了每一条记忆。

- [tasks/outbound_drawers_gate_B.md（文档v1，快照v1，turn=0）](memory/revisions/00000001-40c947e9f009470e903ce5ff9c35ff1c/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v1，快照v1，turn=4）](memory/revisions/00000001-40c947e9f009470e903ce5ff9c35ff1c/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v1，快照v1，turn=16）](memory/revisions/00000001-40c947e9f009470e903ce5ff9c35ff1c/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v2，快照v2，turn=19）](memory/revisions/00000002-2991c5d07cb64fa295ac8eb75cc6fa13/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v3，快照v3，turn=26）](memory/revisions/00000003-899dec438fc44e7abbdab987f925e8f5/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v4，快照v4，turn=31）](memory/revisions/00000004-7e3c22a9aaca465397de6d90b6fc2d0b/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v5，快照v5，turn=40）](memory/revisions/00000005-06952a025d99439b94dd12a70e5c410d/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v6，快照v6，turn=43）](memory/revisions/00000006-54b3e61ffb3b4f489e1ba9d4eba98c24/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v7，快照v7，turn=56）](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/tasks/outbound_drawers_gate_B.md)

记忆初末快照：`{"initial": {"revision": 0, "scope": "roundtrip-82c2daea3ebf493a825a40d1d3d78685", "files": 0, "processed_event_seq": 0, "latest_event_seq": 0, "acknowledged_event_seqs": [], "pending_event_count": 0, "processed_critical_event_seq": 0, "pending_critical_events": 0, "artifact": "outbound/memory-initial.json"}, "final": {"revision": 9, "scope": "roundtrip-82c2daea3ebf493a825a40d1d3d78685", "files": 5, "processed_event_seq": 39, "latest_event_seq": 39, "acknowledged_event_seqs": [], "pending_event_count": 0, "processed_critical_event_seq": 1, "pending_critical_events": 0, "artifact": "outbound/memory-final.json"}}`。

物理评分：`{"success": true, "goal_kind": "composite", "distance_to_target_center_m": 1.0354994649065987, "distance_error_m": 0.03549946490659872, "heading_error_rad": 0.07782547330447054, "standoff_m": 1.0, "distance_tolerance_m": 0.18, "stopped": true, "speed_mps": 0.002467371057294235, "fallen": false, "collision": false, "collision_physics_steps": 0, "target_xy_m": [5.6, 0.0], "base_xy_m": [4.564500819773008, -0.0007678353217312032], "sim_time_s": 57.88000000002477, "source": "privileged_simulator_evaluator", "cabinet_cleared": true, "passage_crossed": true, "beyond_exit": true, "stages_in_order": true, "events": [{"kind": "cabinet_clearance", "trajectory_index": 993, "fraction": 0.20766021459452852, "sim_time_s": 20.864153204291206, "crossing_y_m": 1.0171490666696599}, {"kind": "passage_crossing", "trajectory_index": 2180, "fraction": 0.11073595975147778, "sim_time_s": 44.60221471920357, "crossing_y_m": 0.08998893897822045}, {"kind": "passage_exit", "trajectory_index": 2416, "fraction": 0.9037946180757912, "sim_time_s": 49.33807589237585, "crossing_y_m": -0.0022050902308504406}], "cabinet_exit_x_m": 2.3, "gate_x_m": 3.4, "finish_x_min_m": 4.2, "stages": {"avoid_drawers": true, "pass_gate": true, "approach_b": true}, "stage_completion_rate": 1.0}`。

工具错误：`[{"role": "planner", "turn": 17, "attempt": null, "tool": "turn", "code": "MEMORY_CHANGED"}, {"role": "planner", "turn": 24, "attempt": null, "tool": "turn", "code": "MEMORY_CHANGED"}, {"role": "planner", "turn": 29, "attempt": null, "tool": "move_relative", "code": "MEMORY_CHANGED"}, {"role": "planner", "turn": 38, "attempt": null, "tool": "turn", "code": "MEMORY_CHANGED"}, {"role": "planner", "turn": 41, "attempt": null, "tool": "turn", "code": "MEMORY_CHANGED"}, {"role": "planner", "turn": 47, "attempt": null, "tool": "move_relative", "code": "INVALID_ARGUMENTS"}, {"role": "planner", "turn": 54, "attempt": null, "tool": "turn", "code": "MEMORY_CHANGED"}, {"role": "memory", "turn": null, "attempt": 7, "tool": "memory_read", "code": "ONE_TOOL_PER_TURN"}, {"role": "memory", "turn": null, "attempt": 7, "tool": "memory_read", "code": "ONE_TOOL_PER_TURN"}, {"role": "memory", "turn": null, "attempt": 7, "tool": "memory_read", "code": "ONE_TOOL_PER_TURN"}, {"role": "memory", "turn": null, "attempt": 19, "tool": "commit_memory", "code": "BODY_TOO_LONG"}, {"role": "memory", "turn": null, "attempt": 20, "tool": "commit_memory", "code": "BODY_TOO_LONG"}, {"role": "memory", "turn": null, "attempt": 27, "tool": "commit_memory", "code": "BODY_TOO_LONG"}, {"role": "memory", "turn": null, "attempt": 28, "tool": "commit_memory", "code": "BODY_TOO_LONG"}, {"role": "memory", "turn": null, "attempt": 34, "tool": "memory_read", "code": "ONE_TOOL_PER_TURN"}, {"role": "memory", "turn": null, "attempt": 34, "tool": "memory_read", "code": "ONE_TOOL_PER_TURN"}]`。

## 返程：读取去程记忆并原路返回

- 规划 Agent：请求38，完整回复38，未知用量尝试0；已知用量 `{"prompt_tokens": 702522, "completion_tokens": 46738, "total_tokens": 749260}`。
  请求错误：`[]`；worker 生命周期错误：`[]`。
- 记忆 Agent：请求0，完整回复0，未知用量尝试0；已知用量 `{}`。
  请求错误：`[]`；worker 生命周期错误：`[]`。

模型等待：299.50秒；记忆等待：0.00秒；工具执行墙钟：4.13秒。

实际成功读取 MD 13次；首个成功运动前已有读取：True。这核对读工具记录，不自动证明路线语义正确或规划确实采用了每一条记忆。

- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=0）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)
- [episodes/outbound_01_measurements.md（文档v7，快照v9，turn=1）](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/episodes/outbound_01_measurements.md)
- [episodes/outbound_02_crossing_and_approach.md（文档v8，快照v9，turn=2）](memory/revisions/00000008-2cdda31722614671b711599739c983d9/episodes/outbound_02_crossing_and_approach.md)
- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=3）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=8）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)
- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=12）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)
- [episodes/outbound_01_measurements.md（文档v7，快照v9，turn=14）](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/episodes/outbound_01_measurements.md)
- [MEMORY.md（文档v9，快照v9，turn=18）](memory/revisions/00000009-44493da2390442e398900d2422161c72/MEMORY.md)
- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=19）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)
- [MEMORY.md（文档v9，快照v9，turn=24）](memory/revisions/00000009-44493da2390442e398900d2422161c72/MEMORY.md)
- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=26）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)
- [episodes/outbound_01_measurements.md（文档v7，快照v9，turn=29）](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/episodes/outbound_01_measurements.md)
- [tasks/outbound_drawers_gate_B.md（文档v9，快照v9，turn=35）](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)

记忆初末快照：`{"initial": {"revision": 9, "scope": "roundtrip-82c2daea3ebf493a825a40d1d3d78685", "files": 5, "processed_event_seq": 39, "latest_event_seq": 39, "acknowledged_event_seqs": [], "pending_event_count": 0, "processed_critical_event_seq": 1, "pending_critical_events": 0, "artifact": "return/memory-initial.json"}, "final": {"revision": 9, "scope": "roundtrip-82c2daea3ebf493a825a40d1d3d78685", "files": 5, "processed_event_seq": 39, "latest_event_seq": 39, "acknowledged_event_seqs": [], "pending_event_count": 0, "processed_critical_event_seq": 1, "pending_critical_events": 0, "artifact": "return/memory-final.json"}}`。

返程评分直接取自独立评估器，以下不是从 MD 推测的成绩：

```json
{
  "success": true,
  "geometry_success": true,
  "returned_to_start": true,
  "state_valid": true,
  "return_distance_to_start_m": 0.17149581340060271,
  "return_tolerance_m": 0.3,
  "retrace_tolerance_m": 0.5,
  "outbound_path_length_m": 6.730193976972153,
  "return_path_length_m": 6.5044906977492865,
  "retrace_max_deviation_m": 0.16698915181494436,
  "retrace_mean_deviation_m": 0.08669466887895494,
  "retrace_p95_deviation_m": 0.15403296023392743,
  "outbound_coverage": 1.0,
  "return_corridor_fraction": 1.0,
  "outbound_stages_in_order": true,
  "return_stages_in_order": true,
  "same_cabinet_side": true,
  "retrace_corridor_passed": true,
  "boundaries_continuous": true,
  "invalid_path": false,
  "invalid_samples": 0,
  "heading_required": false,
  "stopped": true,
  "speed_mps": 0.0034399684896449373,
  "yaw_rate_radps": 0.01386964016464147,
  "fallen": false,
  "collision": false,
  "collision_physics_steps": 0,
  "source": "privileged_roundtrip_evaluator",
  "events": [
    {
      "leg": "outbound",
      "kind": "cabinet",
      "direction": "forward",
      "trajectory_index": 815,
      "fraction": 0.9982318464676286,
      "sim_time_s": 17.299964636930643,
      "crossing_xy_m": [
        1.65,
        0.9719049563383555
      ],
      "side": "positive_y"
    },
    {
      "leg": "outbound",
      "kind": "gate",
      "direction": "forward",
      "trajectory_index": 2181,
      "fraction": 0.11073595975147778,
      "sim_time_s": 44.60221471920357,
      "crossing_xy_m": [
        3.4,
        0.08998893897822045
      ],
      "inside_opening": true
    },
    {
      "leg": "return",
      "kind": "gate",
      "direction": "reverse",
      "trajectory_index": 1114,
      "fraction": 0.45234032272369673,
      "sim_time_s": 80.14904680644908,
      "crossing_xy_m": [
        3.4,
        0.035950913011490525
      ],
      "inside_opening": true
    },
    {
      "leg": "return",
      "kind": "cabinet",
      "direction": "reverse",
      "trajectory_index": 2291,
      "fraction": 0.6907845132860526,
      "sim_time_s": 103.69381569020545,
      "crossing_xy_m": [
        1.65,
        0.8176248084001129
      ],
      "side": "positive_y"
    }
  ]
}
```

coverage、轨迹偏离、过门次序、绕柜侧和回到起点均按评估器记录解释；缺失字段表示未记录，不按通过处理。

工具错误：`[{"role": "planner", "turn": 15, "attempt": null, "tool": "move_relative", "code": "INVALID_ARGUMENTS"}]`。

## 记忆与一致性

检查结果：`{"same_physical_boundary": true, "return_memory_unchanged": true, "fresh_return_history": true, "shared_coordinate_frame": true, "same_simulation_clock": true, "return_read_memory": true, "source_frozen": true}`。

- [MD v0](memory/revisions/00000000-067f230967174f62acd3a013014a34a3/manifest.json)：连续水位0，稀疏ACK=[]；没有新增或修订正文。
- [MD v1](memory/revisions/00000001-40c947e9f009470e903ce5ff9c35ff1c/manifest.json)：连续水位1，稀疏ACK=[]；[MEMORY.md](memory/revisions/00000001-40c947e9f009470e903ce5ff9c35ff1c/MEMORY.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000001-40c947e9f009470e903ce5ff9c35ff1c/tasks/outbound_drawers_gate_B.md)。
- [MD v2](memory/revisions/00000002-2991c5d07cb64fa295ac8eb75cc6fa13/manifest.json)：连续水位9，稀疏ACK=[]；[episodes/outbound_01_measurements.md](memory/revisions/00000002-2991c5d07cb64fa295ac8eb75cc6fa13/episodes/outbound_01_measurements.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000002-2991c5d07cb64fa295ac8eb75cc6fa13/tasks/outbound_drawers_gate_B.md)。
- [MD v3](memory/revisions/00000003-899dec438fc44e7abbdab987f925e8f5/manifest.json)：连续水位15，稀疏ACK=[]；[episodes/outbound_01_measurements.md](memory/revisions/00000003-899dec438fc44e7abbdab987f925e8f5/episodes/outbound_01_measurements.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000003-899dec438fc44e7abbdab987f925e8f5/tasks/outbound_drawers_gate_B.md)。
- [MD v4](memory/revisions/00000004-7e3c22a9aaca465397de6d90b6fc2d0b/manifest.json)：连续水位19，稀疏ACK=[]；[episodes/outbound_01_measurements.md](memory/revisions/00000004-7e3c22a9aaca465397de6d90b6fc2d0b/episodes/outbound_01_measurements.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000004-7e3c22a9aaca465397de6d90b6fc2d0b/tasks/outbound_drawers_gate_B.md)。
- [MD v5](memory/revisions/00000005-06952a025d99439b94dd12a70e5c410d/manifest.json)：连续水位21，稀疏ACK=[]；[episodes/outbound_01_measurements.md](memory/revisions/00000005-06952a025d99439b94dd12a70e5c410d/episodes/outbound_01_measurements.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000005-06952a025d99439b94dd12a70e5c410d/tasks/outbound_drawers_gate_B.md)。
- [MD v6](memory/revisions/00000006-54b3e61ffb3b4f489e1ba9d4eba98c24/manifest.json)：连续水位25，稀疏ACK=[]；[episodes/outbound_01_measurements.md](memory/revisions/00000006-54b3e61ffb3b4f489e1ba9d4eba98c24/episodes/outbound_01_measurements.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000006-54b3e61ffb3b4f489e1ba9d4eba98c24/tasks/outbound_drawers_gate_B.md)。
- [MD v7](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/manifest.json)：连续水位27，稀疏ACK=[]；[episodes/outbound_01_measurements.md](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/episodes/outbound_01_measurements.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000007-fc60e571604a4b2499b6794da0b2560d/tasks/outbound_drawers_gate_B.md)。
- [MD v8](memory/revisions/00000008-2cdda31722614671b711599739c983d9/manifest.json)：连续水位35，稀疏ACK=[]；[MEMORY.md](memory/revisions/00000008-2cdda31722614671b711599739c983d9/MEMORY.md)、[episodes/outbound_02_crossing_and_approach.md](memory/revisions/00000008-2cdda31722614671b711599739c983d9/episodes/outbound_02_crossing_and_approach.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000008-2cdda31722614671b711599739c983d9/tasks/outbound_drawers_gate_B.md)。
- [MD v9](memory/revisions/00000009-44493da2390442e398900d2422161c72/manifest.json)：连续水位39，稀疏ACK=[]；[MEMORY.md](memory/revisions/00000009-44493da2390442e398900d2422161c72/MEMORY.md)、[episodes/outbound_03_finish_return_start.md](memory/revisions/00000009-44493da2390442e398900d2422161c72/episodes/outbound_03_finish_return_start.md)、[tasks/outbound_drawers_gate_B.md](memory/revisions/00000009-44493da2390442e398900d2422161c72/tasks/outbound_drawers_gate_B.md)。

连续水位与稀疏ACK分别表示连续已处理前缀和越过积压后单独确认的事件；文件发布和证据引用不保证语义完全正确。

未运行阶段：无；返程未运行原因：`None`。

实验错误：`null`；清理错误：`[]`；计数错误：`[]`。

本轮没有自动获得无记忆对照；即便往返成功，也只证明本次配置完成了任务，不能单凭一次成功证明记忆带来的提升。
