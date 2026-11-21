import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from PIL import ImageDraw

from go2_demo.occluded_video import OccludedRecording, _ffmpeg, render


def make_capture(tmp_path, handoff, return_label=None):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="scenario_target" type="sphere" pos="6.4 0 0" size=".1"/>
          <geom name="memory_goal_b" type="box" pos="1.8 .6 0" size=".1 .1 .1"/>
          <body><freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    data.time = 1.
    backend = SimpleNamespace(model=model, data=data, _recording=None)
    capture = OccludedRecording(backend, tmp_path / "video", return_label=return_label)
    for t in [1.04, 1.1, 1.16, 1.20, 1.28]:
        data.time = t
        data.qpos[0] = t
        capture.sample()
        if t == handoff:
            capture.mark_handoff()
    capture.finish()
    return capture.directory


@pytest.mark.parametrize("handoff,expected", [(1.1, [1.1, 1.16, 1.2, 1.28]), (1.28, [1.28])])
@pytest.mark.parametrize("return_label", [None, "MODEL ROUTE | ODOMETRY CONTROLLER"])
def test_only_return_poses_are_encoded_and_full_manifest_is_preserved(
        tmp_path, monkeypatch, handoff, expected, return_label):
    try:
        _ffmpeg()
    except (ImportError, RuntimeError):
        pytest.skip("ffmpeg encoder unavailable")
    capture = make_capture(tmp_path, handoff, return_label)
    assert json.loads((capture / "capture.json").read_text()).get("return_label") == return_label
    rendered = []
    captions = []
    draw_text = ImageDraw.ImageDraw.text

    def record_caption(draw, xy, text, **kwargs):
        captions.append(text)
        return draw_text(draw, xy, text, **kwargs)

    class TestRenderer:
        def __init__(self, model, height, width):
            self.image = np.zeros((height, width, 3), dtype=np.uint8)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def update_scene(self, data, camera):
            rendered.append(data.time)

        def render(self):
            return self.image.copy()

    monkeypatch.setattr(mujoco, "Renderer", TestRenderer)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_caption)
    return_caption = "RETURN TO B  |  " + (return_label or "LIVE AGENT")
    full = render(capture)
    assert rendered[0] == 1. and rendered[-1] == 1.28
    for t, caption in zip(rendered, captions, strict=True):
        assert caption.startswith(return_caption if t >= handoff else "OUTBOUND TO A  |  SHARED RECORDED ACTIONS")
    rendered.clear()
    captions.clear()
    clip = render(capture, return_only=True)
    assert rendered == expected
    assert all(c.startswith(return_caption) and "return +" in c for c in captions)
    assert clip["frame_count"] == len(expected)
    assert clip["clip"]["simulation_duration_s"] == pytest.approx(expected[-1] - expected[0])
    assert clip["decoded_all_frames"]
    assert (capture / "return.mp4").is_file()
    assert json.loads((capture / "return-video.json").read_text()) == clip
    assert json.loads((capture / "video.json").read_text()) == full


@pytest.mark.parametrize("handoff,error", [(None, "No A handoff"), (1.18, "does not match")])
def test_requires_an_exact_recorded_handoff(tmp_path, handoff, error):
    capture = make_capture(tmp_path, handoff=None)
    path = capture / "capture.json"
    meta = json.loads(path.read_text())
    meta["handoff_s"] = handoff
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=error):
        render(capture, return_only=True)
    assert not (capture / "return.mp4").exists()
