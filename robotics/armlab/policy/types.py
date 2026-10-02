"""Data passed between an environment (MuJoCo here, RoboLab on GPU) and a policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

Gripper = Literal["open", "close"]


@dataclass
class Waypoint:
    """An end-effector target: fingertip-center position (m, robot base frame), yaw (deg), gripper."""
    xyz: tuple[float, float, float]
    yaw_deg: float = 0.0
    gripper: Gripper | None = None  # applied after reaching xyz; None = leave as is

    def to_dict(self) -> dict:
        return {"xyz": [round(float(v), 4) for v in self.xyz], "yaw_deg": round(float(self.yaw_deg), 1), "gripper": self.gripper}

    @classmethod
    def from_dict(cls, d: dict) -> "Waypoint":
        xyz = d.get("xyz") or [d["x"], d["y"], d["z"]]
        if len(xyz) != 3:
            raise ValueError(f"xyz must have 3 numbers, got {xyz}")
        g = d.get("gripper")
        if g not in (None, "open", "close"):
            raise ValueError(f"gripper must be 'open', 'close' or null, got {g!r}")
        return cls(tuple(float(v) for v in xyz), float(d.get("yaw_deg", 0.0)), g)


@dataclass
class Observation:
    instruction: str
    images: dict[str, np.ndarray]  # camera name -> HxWx3 uint8
    eef_xyz: np.ndarray
    eef_yaw_deg: float
    gripper_width: float  # meters between fingers (0 .. 0.08)
    step: int
    max_steps: int
    sim_time: float
    task_done: bool = False  # the environment's success signal ("simulator terminated")
    objects: dict[str, dict] | None = None  # privileged state, only in obs_mode="state"
    pixel_to_xyz: Callable[[str, int, int], np.ndarray | None] | None = None  # depth lookup tool
    xyz_to_pixel: Callable[[str, object], tuple[int, int] | None] | None = None  # projection (for overlays)
    feedback: str = ""  # what the harness reports about the previous segment
    camera_info: dict[str, str] = field(default_factory=dict)  # camera name -> description


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.latency_s += other.latency_s
        self.calls += other.calls


@dataclass
class Decision:
    waypoints: list[Waypoint]
    source: str  # "system1" | "llm" | "llm-correction" | "oracle"
    rationale: str = ""
    done: bool = False  # policy believes the task is complete
    usage: Usage = field(default_factory=Usage)
    candidate: list[Waypoint] | None = None  # hybrid: the System-1 proposal that was reviewed
    notes: str = ""  # persistent notes the policy wants to carry forward


class Policy:
    name = "policy"

    def reset(self, instruction: str) -> None:  # pragma: no cover - interface
        pass

    def act(self, obs: Observation) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError
