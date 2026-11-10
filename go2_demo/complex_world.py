"""Fixed multi-zone Go2 worlds and private, measured route acceptance.

Scene geometry never depends on an order. Route permissions are evaluated only
after execution; they do not select which doors the robot can see or traverse.
"""

from __future__ import annotations

import math
from numbers import Real

from .scenarios import COLORS, SceneObject, Scenario, TargetSpec, get_scenario


DESTINATION_XY = {"A": (7.2, 1.4), "B": (7.2, 0.0), "C": (7.2, -1.4)}
FIRST_GATE_X = 2.6
GREEN_GATE_X = 5.2
EXIT_X = 6.0
ROUTE_CENTER_Y = {"blue": 1.25, "yellow": -1.25}
ROUTE_HALF_WIDTH = 0.55
_GRAY = (0.32, 0.36, 0.42, 1.0)
_WOOD = (0.55, 0.33, 0.14, 1.0)


def _box(name, xyz, size, color=_GRAY, role="obstacle"):
    return SceneObject(name, "box", tuple(xyz), tuple(size), color, role)


def _gate(name, x, center_y, color, *, post_offset=0.8):
    return (
        _box(name + "_left", (x, center_y + post_offset, 0.65), (.1, .1, .65), COLORS[color]),
        _box(name + "_right", (x, center_y - post_offset, 0.65), (.1, .1, .65), COLORS[color]),
        _box(name + "_top", (x, center_y, 1.35), (.1, post_offset + .1, .08), COLORS[color]),
    )


def _signs(destinations):
    objects = [SceneObject("sign_" + letter, "sign", (*xy, .65),
                           (.025, .20, .25), COLORS["white"], "distractor", letter=letter)
               for letter, xy in destinations.items() if letter != "B"]
    objects.extend(_box("sign_support_" + letter, (*xy, .20), (.025, .025, .20))
                   for letter, xy in destinations.items())
    return tuple(objects)


def final_scenario() -> Scenario:
    """Two open alternative doors, a merge obstacle, then a common green door."""
    instruction = (
        "这是多区交付场景：出发区有抽屉柜；中间隔墙设有蓝色和黄色两扇门，两扇门都能通行。"
        "只允许使用交班记录中当前有效的授权通道，不得穿越另一扇门或绕过隔墙。"
        "先绕过出发区柜体，从授权的蓝门或黄门开口正向穿过；"
        "再绕开汇合区的第二个柜体，从绿色门框开口正向穿过，"
        "机身base原点穿过绿门后须沿穿门方向继续至少0.8米。"
        "最后在A、B、C三块白色标牌中选择当前订单的目的地，"
        "移动到目标中心水平距离约1米、面向目标并停稳。"
        "所有门均用机身base原点判定穿越；穿越蓝门、黄门或绿门平面时，"
        "base原点相对该开口中心线的横向偏差必须不超过0.55米，以保留机身余量。"
        "前置相机位于base前方0.32米，门框消失不代表机身已经穿过。"
        "结合可见表面选点、local_odometry和实际运动反馈确认位置，不能累加请求距离代替实测位移。"
        "全程不得接触柜体、墙、门框、标牌或其他物体；正确通道、目的地及交付信息由任务和记忆确定，"
        "场景颜色与几何不会替你标出当前有效订单。完成任务要求的提交后调用finish。"
    )
    objects = [
        _box("dispatch_cabinet", (1.25, 0., .30), (.25, .42, .30), _WOOD),
        _box("merge_cabinet", (3.7, -.30, .30), (.28, .45, .30), _WOOD),
        _box("north_wall", (3.5, 3.05, .65), (4.4, .05, .65)),
        _box("south_wall", (3.5, -3.05, .65), (4.4, .05, .65)),
        _box("back_wall", (-.9, 0., .65), (.05, 3.10, .65)),
        _box("front_wall", (7.9, 0., .65), (.05, 3.10, .65)),
    ]
    for i, (lo, hi) in enumerate(((-3., -2.15), (-.35, .35), (2.15, 3.))):
        objects.append(_box("divider_" + str(i), (FIRST_GATE_X, (lo + hi) / 2, .65),
                            (.1, (hi - lo) / 2, .65)))
    for color, center in ROUTE_CENTER_Y.items():
        objects.extend(_gate(color + "_gate", FIRST_GATE_X, center, color))
    objects.extend(_gate("green_gate", GREEN_GATE_X, 0., "green", post_offset=.9))
    objects.extend(_signs(DESTINATION_XY))
    return Scenario(
        task_id="memory_complex_multizone", title="多区分岔、绕障与交付", category="memory_complex",
        instruction=instruction,
        target=TargetSpec("sign", "white", DESTINATION_XY["B"], z=.65,
                          half_size=(.025, .20, .25), letter="B"),
        objects=tuple(objects), goal_kind="standoff", target_jitter_m=0.,
        source="local_complex_memory_mujoco",
    )


def learning_scenario(stage: str) -> Scenario:
    """Separate local-coordinate learning areas; every call starts a new episode."""
    if stage == "dispatch":
        return get_scenario("composite_drawers_gate_b")
    if stage == "fragments":
        target_x = 4.2
        objects = (
            _box("first_offset_crate", (1.15, -.40, .28), (.20, .28, .28), _WOOD),
            _box("second_offset_crate", (2.50, .40, .28), (.20, .28, .28), (0.35, .42, .50, 1.)),
        )
        title, description = "分片登记区交错绕障", "绕过前后错位的两个箱体"
    elif stage == "revision":
        target_x = 3.5
        objects = (
            _box("maintenance_equipment", (1.55, -.12, .30), (.28, .40, .30), _GRAY),
            _box("equipment_top", (1.55, -.12, .62), (.30, .42, .02), (0.06, .07, .08, 1.)),
        )
        title, description = "变更登记区设备绕障", "绕过前方的维护设备"
    else:
        raise ValueError("Unknown learning stage; expected dispatch, fragments or revision")
    objects += (_box("sign_support_B", (target_x, 0., .20), (.025, .025, .20)),)
    instruction = (description + "，找到其后的白色字母B标牌，移动到距离B中心水平距离约1米处，"
                   "面向B并停稳，然后调用finish。全程不得接触任何箱体、设备、支架或标牌。"
                   "这是独立学习区域；本阶段的B只是巡检终点，不代表后续订单目的地。")
    return Scenario("memory_complex_" + stage, title, "memory_complex", instruction,
                    TargetSpec("sign", "white", (target_x, 0.), z=.65,
                               half_size=(.025, .20, .25), letter="B"), objects,
                    goal_kind="standoff", target_jitter_m=0., source="local_complex_memory_mujoco")


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def evaluate_route(trajectory, route, start_xy=(0., 0.)) -> dict:
    """Accept only an observed permitted-door → green-door → clearance chain.

    Every crossing interpolates adjacent finite, time-ordered samples. Any wrong
    or outside first-divider crossing is permanent failure, including reverse
    travel. Returning before clearance requires a fresh ordered crossing chain.
    Invalid samples cannot bridge a gap or become a successful route.
    """
    if route not in ROUTE_CENTER_Y:
        raise ValueError("Route must be blue or yellow")
    if (not isinstance(start_xy, (list, tuple)) or len(start_xy) != 2
            or not all(_finite(v) for v in start_xy)):
        raise ValueError("start_xy must contain two finite coordinates")
    if not isinstance(trajectory, list):
        raise ValueError("trajectory must be a list")
    previous, previous_time = tuple(start_xy), None
    first_gate, first_active, green_active, cleared = None, False, False, False
    wrong, outside, green_outside, disorder, invalid = 0, 0, 0, 0, 0
    events = []
    final_xy = None
    for index, row in enumerate(trajectory):
        xyz = row.get("base_xyz_m") if isinstance(row, dict) else None
        timestamp = row.get("sim_time_s") if isinstance(row, dict) else None
        if (not isinstance(xyz, (list, tuple)) or len(xyz) != 3
                or not all(_finite(v) for v in xyz) or not _finite(timestamp)
                or previous_time is not None and timestamp <= previous_time):
            invalid += 1
            previous = previous_time = final_xy = None
            first_active = green_active = False
            continue
        point = (float(xyz[0]), float(xyz[1]))
        final_xy = point
        if previous is not None and point[0] != previous[0]:
            forward = point[0] > previous[0]
            planes = (FIRST_GATE_X, GREEN_GATE_X, EXIT_X)
            for plane in planes if forward else reversed(planes):
                crossed = (previous[0] < plane <= point[0] if forward else point[0] < plane <= previous[0])
                if not crossed:
                    continue
                fraction = (plane - previous[0]) / (point[0] - previous[0])
                y = previous[1] + fraction * (point[1] - previous[1])
                when = (previous_time + fraction * (timestamp - previous_time)
                        if previous_time is not None else None)
                if plane == FIRST_GATE_X:
                    gate = next((name for name, center in ROUTE_CENTER_Y.items()
                                 if abs(y - center) <= ROUTE_HALF_WIDTH), "outside")
                    if first_gate is None:
                        first_gate = gate
                    outside += int(gate == "outside")
                    wrong += int(gate not in {route, "outside"})
                    if forward and gate == route:
                        first_active = True
                    elif not forward:
                        first_active = green_active = False
                elif plane == GREEN_GATE_X:
                    gate = "green" if abs(y) <= ROUTE_HALF_WIDTH else "green_outside"
                    green_outside += int(gate == "green_outside")
                    if forward and gate == "green":
                        if first_active:
                            green_active = True
                        else:
                            disorder += 1
                    elif not forward:
                        green_active = False
                else:
                    gate = "clearance"
                    if forward and first_active and green_active:
                        cleared = True
                events.append({"gate": gate, "direction": "forward" if forward else "reverse",
                               "trajectory_index": index, "fraction": float(fraction),
                               "sim_time_s": float(when) if when is not None else None,
                               "crossing_xy_m": [float(plane), float(y)]})
        previous, previous_time = point, float(timestamp)
    beyond = bool(final_xy is not None and final_xy[0] >= EXIT_X)
    success = bool(trajectory and cleared and beyond and first_active and green_active
                   and not (wrong or outside or green_outside or disorder or invalid))
    return {"success": success, "route": route, "ordered_gate_events": events,
            "first_gate": first_gate, "wrong_gate_first": first_gate is not None and first_gate != route,
            "wrong_gate_crossings": wrong, "first_barrier_outside_crossings": outside,
            "green_outside_crossings": green_outside, "out_of_order_crossings": disorder,
            "green_crossed": any(e["gate"] == "green" and e["direction"] == "forward" for e in events),
            "green_clearance_reached": cleared, "beyond_green_clearance": beyond,
            "invalid_path": bool(invalid), "invalid_samples": invalid,
            "source": "privileged_measured_trajectory"}
