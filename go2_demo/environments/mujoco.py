"""Build a planar MuJoCo world around a supplied robot XML asset.

The environment owns task geometry and seeded target sampling. Robot controllers
supply their model asset and camera mount; this module does not load a policy.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..scenarios import COLORS, LETTER_PATTERNS


def add_letter(world, name: str, position, size, letter: str) -> None:
    """Place readable black 7x5 glyph cells on the front of a white sign."""
    if letter not in LETTER_PATTERNS:
        raise ValueError(f"Unsupported scene letter: {letter}")
    x, y, z = position
    half_x, half_y, half_z = size
    cell_w, cell_h = 2 * half_y * .14, 2 * half_z * .105
    count = 0
    for row, pattern in enumerate(LETTER_PATTERNS[letter]):
        for column, bit in enumerate(pattern):
            if bit != "1":
                continue
            # Camera forward is +x; increasing image u points toward world -y.
            position = (x - half_x - .003, y - (column - 2) * cell_w, z + (3 - row) * cell_h)
            ET.SubElement(world, "geom", name=f"{name}_glyph_{count}", type="box",
                          pos=" ".join(str(value) for value in position),
                          size=f".002 {cell_w / 2} {cell_h / 2}", rgba=".015 .015 .015 1",
                          contype="0", conaffinity="0")
            count += 1


class MujocoEnvironment:
    """A scene and its physical collision geometries, without action control."""

    def __init__(self, scenario=None, seed: int = 0, *, target_xy=None):
        if scenario is not None and not scenario.supported:
            raise ValueError(f"UNSUPPORTED_CAPABILITY: {scenario.unsupported_reason}")
        self.scenario = scenario
        if target_xy is None:
            rng = np.random.default_rng(seed)
            target_xy = (np.asarray(scenario.target.xy, dtype=float).copy() if scenario is not None else
                         np.array([3.0 + rng.uniform(-0.15, 0.15), 0.55 + rng.uniform(-0.15, 0.15)]))
            if scenario is not None and scenario.target_jitter_m:
                target_xy += rng.uniform(-scenario.target_jitter_m, scenario.target_jitter_m, size=2)
        self._target_xy = np.asarray(target_xy, dtype=float).copy()
        self._environment_geom_names = []

    @property
    def target_xy(self):
        return self._target_xy

    @property
    def collision_geom_names(self):
        return tuple(self._environment_geom_names)

    def make_model(self, robot_xml: Path, *, dt: float, width: int, height: int,
                   camera_body_name: str, camera_attributes: dict):
        self._environment_geom_names = []
        robot_xml = Path(robot_xml)
        source = robot_xml.parent
        root = ET.parse(robot_xml).getroot()
        # Load meshes through MuJoCo's virtual filesystem. This also supports
        # Windows workspaces whose paths contain non-ASCII characters.
        root.find("compiler").set("meshdir", "")
        mesh_assets = {path.name: path.read_bytes() for path in (source / "assets").glob("*.obj")}
        ET.SubElement(root, "option", timestep=str(dt), gravity="0 0 -9.81", integrator="implicitfast")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
        ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.3 0.3 0.3", specular="0.2 0.2 0.2")
        ET.SubElement(visual, "map", znear="0.01", zfar="20")
        asset = root.find("asset")
        ET.SubElement(asset, "texture", name="floor_texture", type="2d", builtin="checker", rgb1="0.28 0.32 0.36", rgb2="0.42 0.46 0.50", width="128", height="128")
        ET.SubElement(asset, "material", name="floor_material", texture="floor_texture", texrepeat="12 12", texuniform="true", reflectance="0")
        world = root.find("worldbody")
        ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8", directional="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 .1", material="floor_material", friction="1 .005 .0001")
        if self.scenario is None:
            target_x, target_y = self._target_xy
            ET.SubElement(world, "geom", name="red_target", type="box", pos=f"{target_x} {target_y} .5", size=".08 .20 .5", rgba=".85 .035 .025 1", friction="1 .005 .0001")
            # A small white cross makes the red target center a genuine visual feature.
            # These decorations have no physical collision geometry.
            ET.SubElement(world, "geom", name="target_cross_v", type="box", pos=f"{target_x-.082} {target_y} .5", size=".002 .018 .075", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
            ET.SubElement(world, "geom", name="target_cross_h", type="box", pos=f"{target_x-.083} {target_y} .5", size=".002 .075 .018", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
            ET.SubElement(world, "geom", name="blue_distractor", type="box", pos="3 -1.3 .4", size=".12 .18 .4", rgba=".04 .12 .85 1")
            self._environment_geom_names = ["red_target", "blue_distractor"]
        else:
            self._populate_scenario(world)
        base = world.find(f"body[@name='{camera_body_name}']")
        if base is None:
            raise ValueError("Robot camera body is missing from the model")
        ET.SubElement(base, "camera", **camera_attributes)
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=mesh_assets)

    def _populate_scenario(self, world) -> None:
        target = self.scenario.target
        color = COLORS[target.color] if isinstance(target.color, str) else target.color
        position = (*self._target_xy, target.z)
        ET.SubElement(world, "geom", name="scenario_target", type="box" if target.kind in {"marker", "sign"} else target.kind,
                      pos=" ".join(str(value) for value in position),
                      size=" ".join(str(value) for value in target.half_size),
                      rgba=" ".join(str(value) for value in color), friction="1 .005 .0001")
        self._environment_geom_names.append("scenario_target")
        if target.letter:
            add_letter(world, "scenario_target", position, target.half_size, target.letter)
        elif self.scenario.category in {"baseline", "go_avoid", "go_through"}:
            front = self._target_xy[0] - target.half_size[0] - .002
            y, z = position[1:]
            ET.SubElement(world, "geom", name="target_cross_v", type="box", pos=f"{front} {y} {z}",
                          size=".002 .018 .075", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
            ET.SubElement(world, "geom", name="target_cross_h", type="box", pos=f"{front - .001} {y} {z}",
                          size=".002 .075 .018", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
        for item in self.scenario.objects:
            attributes = {"name": item.name, "type": "box" if item.kind == "sign" else item.kind,
                          "pos": " ".join(str(value) for value in item.position),
                          "size": " ".join(str(value) for value in item.size),
                          "rgba": " ".join(str(value) for value in item.rgba), "friction": "1 .005 .0001"}
            if item.euler is not None:
                attributes["euler"] = " ".join(str(value) for value in item.euler)
            if item.role == "decoration":
                attributes.update(contype="0", conaffinity="0")
            else:
                self._environment_geom_names.append(item.name)
            ET.SubElement(world, "geom", **attributes)
            if item.letter:
                add_letter(world, item.name, item.position, item.size, item.letter)

