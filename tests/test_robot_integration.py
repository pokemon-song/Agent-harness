"""Integration contracts for robot selection, model context and experiment identity."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from go2_demo import __main__ as cli
from go2_demo.models import APIModel, build_system_prompt
from go2_demo.provenance import source_hashes
from go2_demo.robots.go2.profile import GO2_PROFILE
from go2_demo.scenarios import bind_robot, get_scenario


@pytest.mark.parametrize("option", ["--list-robots", "--describe-robot", "--list-tasks"])
def test_inspection_does_not_import_simulator_or_model_transport(option):
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'numpy', 'httpx', 'cv2'}:
            raise AssertionError('Inspection imported runtime dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from go2_demo.__main__ import main
raise SystemExit(main([sys.argv[1]]))
"""
    result = subprocess.run([sys.executable, "-c", script, option], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "go2" in result.stdout.lower() or "go_to_blue_ball" in result.stdout


@pytest.mark.parametrize("args", [["--robot", "unknown"], ["--backend", "hardware"]])
def test_unknown_robot_or_backend_is_rejected_before_credentials(monkeypatch, args):
    monkeypatch.setattr(cli, "_run_single", lambda *a: pytest.fail("unimplemented robot was executed"))
    monkeypatch.setattr(cli, "_make_model", lambda *a: pytest.fail("credentials were loaded"))
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_profile_description_is_serialized_without_runtime(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_make_model", lambda *a: pytest.fail("credentials were loaded"))
    assert cli.main(["--robot", "go2", "--backend", "mujoco", "--describe-robot"]) == 0
    assert json.loads(capsys.readouterr().out) == GO2_PROFILE.to_dict()


def test_task_binding_preserves_go2_and_uses_other_camera_mounting():
    original = get_scenario("composite_drawers_gate_b")
    assert bind_robot(original, GO2_PROFILE).instruction == original.instruction
    other = replace(GO2_PROFILE, robot_id="fixture", sensor=replace(
        GO2_PROFILE.sensor, camera_position_in_body_m=(-.2, 0., .1)))
    bound = bind_robot(original, other)
    assert "后方0.2米" in bound.instruction and "前方0.32米" not in bound.instruction
    assert bind_robot(bound, GO2_PROFILE).instruction == original.instruction
    assert bound.target == original.target and bound.objects == original.objects


def test_task_needing_depth_is_ineligible_on_rgb_only_robot():
    rgb_only = replace(GO2_PROFILE, point_tracking=False,
                       sensor=replace(GO2_PROFILE.sensor, depth_enabled=False))
    result = bind_robot(get_scenario("go_avoid_texture_drawers"), rgb_only)
    assert not result.supported
    assert "depth" in result.unsupported_reason
    assert bind_robot(result, GO2_PROFILE).supported
    # A task with an intrinsic limitation stays disabled after rebinding.
    crawl = bind_robot(get_scenario("crawl_red_gate2"), rgb_only)
    assert not bind_robot(crawl, GO2_PROFILE).supported


def test_model_sends_bound_robot_prompt_in_actual_wire_request():
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "tool_calls": [{"id": "one", "type": "function", "function": {
                "name": "observe", "arguments": "{}"}}]}}]})

    model = APIModel("https://example.test/v1", "mock", "mock-key",
                     transport=httpx.MockTransport(respond))
    profile = replace(GO2_PROFILE, robot_id="test-base", display_name="Test mobile base",
                      height_pitch_control=False, posture=None)
    model.system_prompt = build_system_prompt(profile)
    tools = [{"type": "function", "function": {"name": "observe", "parameters": {
        "type": "object", "properties": {}, "required": []}}}]
    model.decide({}, "data:image/png;base64,AA==", tools, [])
    model.close()
    assert seen[0]["messages"][0]["content"] == model.system_prompt
    assert "test-base" in model.system_prompt and "Unitree Go2" not in model.system_prompt
    assert seen[0]["tools"] == tools


def test_source_fingerprint_covers_nested_robot_and_environment_modules():
    root = Path(__file__).resolve().parents[1]
    hashes = source_hashes(root)
    for name in ("go2_demo/robots/go2/profile.py", "go2_demo/robots/go2/backend.py",
                 "go2_demo/environments/mujoco.py", "go2_demo/task_evaluation.py"):
        assert name in hashes and len(hashes[name]) == 64
