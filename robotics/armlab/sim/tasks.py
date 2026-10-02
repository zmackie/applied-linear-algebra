"""Tabletop tasks in the spirit of the RoboLab subset used in the GPT-6 Astra report.

Each task samples a seeded layout and defines success plus a partial-progress score
(the RoboDojo-style "Score"). The robot base sits at the origin facing +x; the robot's
left is +y. Tables are the floor plane (z = 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .env import ArmEnv

COLORS = {
    "red": (0.85, 0.12, 0.12, 1),
    "green": (0.15, 0.7, 0.2, 1),
    "blue": (0.15, 0.3, 0.9, 1),
    "yellow": (0.95, 0.85, 0.1, 1),
    "orange": (0.95, 0.5, 0.1, 1),
    "purple": (0.55, 0.2, 0.75, 1),
}


@dataclass
class BlockSpec:
    name: str
    color: str
    half: float  # half edge length (m)
    xy: tuple[float, float]
    yaw: float = 0.0

    @property
    def rgba(self):
        return COLORS[self.color]


@dataclass
class BinSpec:
    name: str
    rgba: tuple
    xy: tuple[float, float]
    half_xy: tuple[float, float] = (0.085, 0.085)
    height: float = 0.06
    label: str = ""  # how the instruction refers to it


@dataclass
class ConveyorSpec:
    """A mocap plate that carries a block across the workspace at constant speed."""
    start_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    half: tuple[float, float, float] = (0.07, 0.07, 0.01)


@dataclass
class Layout:
    blocks: list[BlockSpec]
    bins: list[BinSpec]
    conveyor: ConveyorSpec | None = None
    conveyor_block: str | None = None


@dataclass
class TaskStatus:
    success: bool
    score: float  # 0..100 partial completion
    info: dict = field(default_factory=dict)


@dataclass
class Task:
    name: str
    instruction: str
    sample: Callable[[np.random.Generator], Layout]
    check: Callable[["ArmEnv"], TaskStatus]
    max_steps: int = 1500  # control steps at 25 Hz (60 s of sim time)
    tags: tuple[str, ...] = ()
    # Ordered (object, bin-or-None, place_xyz-or-None) goals the scripted oracle executes.
    plan: Callable[["ArmEnv"], list[tuple[str, str | None, tuple | None]]] | None = None


def _scatter(rng, n, avoid=(), x=(0.36, 0.62), y=(-0.24, 0.24), min_dist=0.09):
    pts: list[tuple[float, float]] = []
    for _ in range(2000):
        if len(pts) == n:
            break
        p = (rng.uniform(*x), rng.uniform(*y))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > min_dist for q in [*pts, *avoid]):
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError("could not place objects")
    return pts


def _bin_avoid(bins: list[BinSpec]):
    out = []
    for b in bins:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.append((b.xy[0] + dx * b.half_xy[0], b.xy[1] + dy * b.half_xy[1]))
    return out


# ---------------------------------------------------------------------------------------
# Task definitions


def _blocks_into_bin_sample(rng):
    side = rng.choice([-1, 1])
    bins = [BinSpec("bin", (0.55, 0.55, 0.58, 1), (0.5 + rng.uniform(-0.03, 0.03), side * 0.3), label="gray bin")]
    pts = _scatter(rng, 3, avoid=_bin_avoid(bins), y=(-0.2, 0.2) if side < 0 else (-0.2, 0.2))
    blocks = [BlockSpec(f"{c}_block", c, 0.02, p, rng.uniform(-0.35, 0.35)) for c, p in zip(["red", "green", "blue"], pts)]
    return Layout(blocks, bins)


def _all_in(env: "ArmEnv", objs, bin_name) -> TaskStatus:
    done = [o for o in objs if env.in_bin(o, bin_name)]
    return TaskStatus(len(done) == len(objs), 100.0 * len(done) / len(objs), {"in_bin": done})


def _stack_sample(rng):
    pts = _scatter(rng, 3)
    order = list(rng.permutation(["red", "green", "blue"]))
    blocks = [BlockSpec(f"{c}_block", c, 0.02, p, rng.uniform(-0.35, 0.35)) for c, p in zip(order, pts)]
    return Layout(blocks, [])


def _stack_check(env: "ArmEnv") -> TaskStatus:
    order = ["red_block", "green_block", "blue_block"]
    pos = {o: env.object_pos(o) for o in order}
    h = 0.04

    def on(upper, lower):
        d = pos[upper] - pos[lower]
        return bool(np.hypot(d[0], d[1]) < 0.025 and abs(d[2] - h) < 0.012)

    red_on_table = abs(pos["red_block"][2] - h / 2) < 0.01
    green_ok = red_on_table and on("green_block", "red_block")
    blue_ok = green_ok and on("blue_block", "green_block")
    success = blue_ok and not env.held_objects()
    return TaskStatus(success, 50.0 * (green_ok + blue_ok), {"green_on_red": green_ok, "blue_on_green": blue_ok})


def _stack_plan(env):
    base = env.object_pos("red_block")
    return [("green_block", None, (base[0], base[1], 0.06)), ("blue_block", None, (base[0], base[1], 0.10))]


def _larger_sample(rng):
    side = rng.choice([-1, 1])
    bins = [BinSpec("bin", (0.55, 0.55, 0.58, 1), (0.5, side * 0.3), label="gray bin")]
    pts = _scatter(rng, 3, avoid=_bin_avoid(bins))
    sizes = rng.permutation([0.0175, 0.03])
    blocks = [
        BlockSpec("blue_box_a", "blue", float(sizes[0]), pts[0], rng.uniform(-0.3, 0.3)),
        BlockSpec("blue_box_b", "blue", float(sizes[1]), pts[1], rng.uniform(-0.3, 0.3)),
        BlockSpec("yellow_block", "yellow", 0.02, pts[2], rng.uniform(-0.3, 0.3)),
    ]
    return Layout(blocks, bins)


def _larger_name(env):
    return max(("blue_box_a", "blue_box_b"), key=lambda n: env.layout_block(n).half)


def _larger_check(env: "ArmEnv") -> TaskStatus:
    big = _larger_name(env)
    small = "blue_box_b" if big == "blue_box_a" else "blue_box_a"
    big_in = env.in_bin(big, "bin")
    wrong = [n for n in (small, "yellow_block") if env.in_bin(n, "bin")]
    return TaskStatus(big_in and not wrong, 100.0 * (big_in and not wrong), {"big": big, "wrong_in_bin": wrong})


WARM, COOL = ("red", "yellow", "orange"), ("blue", "green", "purple")


def _sort_sample(rng):
    flip = rng.choice([-1, 1])
    bins = [
        BinSpec("white_bin", (0.95, 0.95, 0.95, 1), (0.48, flip * 0.3), label="white bin"),
        BinSpec("black_bin", (0.12, 0.12, 0.12, 1), (0.48, -flip * 0.3), label="black bin"),
    ]
    colors = [rng.choice(WARM), rng.choice(COOL)]
    colors += [rng.choice([c for c in WARM if c != colors[0]]), rng.choice([c for c in COOL if c != colors[1]])]
    pts = _scatter(rng, 4, avoid=_bin_avoid(bins), x=(0.36, 0.64), y=(-0.16, 0.16), min_dist=0.08)
    blocks = [BlockSpec(f"{c}_block", c, 0.02, p, rng.uniform(-0.3, 0.3)) for c, p in zip(colors, pts)]
    return Layout(blocks, bins)


def _sort_target(name: str) -> str:
    return "white_bin" if name.split("_")[0] in WARM else "black_bin"


def _sort_check(env: "ArmEnv") -> TaskStatus:
    names = [b.name for b in env.layout.blocks]
    right = [n for n in names if env.in_bin(n, _sort_target(n))]
    wrong = [n for n in names if env.in_bin(n, "white_bin" if _sort_target(n) == "black_bin" else "black_bin")]
    score = 100.0 * max(0, len(right) - len(wrong)) / len(names)
    return TaskStatus(len(right) == len(names), score, {"correct": right, "wrong": wrong})


def _conveyor_sample(rng):
    bins = [BinSpec("bin", (0.55, 0.55, 0.58, 1), (0.32, -0.32), label="gray bin")]
    start = (0.55 + rng.uniform(-0.03, 0.03), 0.42)
    speed = rng.uniform(0.035, 0.045)
    conv = ConveyorSpec(start, (0.0, -speed))
    blocks = [BlockSpec("orange_block", "orange", 0.02, (start[0], start[1]), 0.0)]
    return Layout(blocks, bins, conveyor=conv, conveyor_block="orange_block")


TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        Task(
            "blocks_into_bin",
            "Put all three blocks into the gray bin.",
            _blocks_into_bin_sample,
            lambda env: _all_in(env, ["red_block", "green_block", "blue_block"], "bin"),
            tags=("pick_place",),
            plan=lambda env: [(n, "bin", None) for n in ("red_block", "green_block", "blue_block")],
        ),
        Task(
            "stack_in_order",
            "Build a tower: the red block on the bottom, the green block on top of it, and the blue block on top.",
            _stack_sample,
            _stack_check,
            tags=("stacking", "procedural"),
            plan=_stack_plan,
        ),
        Task(
            "larger_into_bin",
            "Put the larger of the two blue blocks into the gray bin. Leave everything else on the table.",
            _larger_sample,
            _larger_check,
            tags=("semantic", "relational"),
            plan=lambda env: [(_larger_name(env), "bin", None)],
        ),
        Task(
            "sort_warm_cool",
            "Put the warm-colored blocks (red, orange, yellow) in the white bin and the cool-colored blocks (blue, green, purple) in the black bin.",
            _sort_sample,
            _sort_check,
            max_steps=2000,
            tags=("semantic", "classification"),
            plan=lambda env: [(b.name, _sort_target(b.name), None) for b in env.layout.blocks],
        ),
        Task(
            "conveyor_pick",
            "An orange block is riding a moving conveyor plate. Pick it up before it leaves the workspace and put it in the gray bin.",
            _conveyor_sample,
            lambda env: _all_in(env, ["orange_block"], "bin"),
            max_steps=1000,
            tags=("dynamic", "latency"),
            plan=lambda env: [("orange_block", "bin", None)],
        ),
    ]
}
