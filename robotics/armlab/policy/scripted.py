"""Privileged scripted policies.

`ScriptedPolicy` is an oracle that reads simulator state and runs pick/place primitives.
With `wrong_target_prob` / `xy_noise` it doubles as a stand-in for a learned "System 1"
policy (the role π0.5 plays in the report): fluent motions, imperfect grounding.
"""
from __future__ import annotations

import math

import numpy as np

from ..sim.env import ArmEnv
from .types import Decision, Observation, Policy, Waypoint

CARRY_Z = 0.20


class ScriptedPolicy(Policy):
    name = "scripted"

    def __init__(self, env: ArmEnv, wrong_target_prob: float = 0.0, xy_noise: float = 0.0, seed: int = 0,
                 source: str = "oracle"):
        self.env = env
        self.wrong_target_prob = wrong_target_prob
        self.xy_noise = xy_noise
        self.rng = np.random.default_rng(seed)
        self.source = source
        self._intent: tuple[str, str | None, tuple | None] | None = None

    def reset(self, instruction: str) -> None:
        self._intent = None

    # --------------------------------------------------------------- helpers

    def grasp_yaw_deg(self, obj: str) -> float:
        """Gripper yaw (relative to the ready pose) that aligns the fingers with the block's faces."""
        env = self.env
        finger_axis = env._R_home[:, 1]
        base = math.atan2(finger_axis[1], finger_axis[0])
        rel = env.object_yaw(obj) - base
        rel = (rel + math.pi / 4) % (math.pi / 2) - math.pi / 4
        return math.degrees(rel)

    def _noisy(self, xy):
        if self.xy_noise <= 0:
            return np.asarray(xy, float)
        return np.asarray(xy, float) + self.rng.normal(0, self.xy_noise, 2)

    def _goal_done(self, goal) -> bool:
        obj, bin_name, place = goal
        if bin_name is not None:
            return self.env.in_bin(obj, bin_name)
        p = self.env.object_pos(obj)
        return bool(np.hypot(p[0] - place[0], p[1] - place[1]) < 0.02 and abs(p[2] - place[2]) < 0.012
                    and obj not in self.env.held_objects())

    def _maybe_corrupt(self, goal):
        """Simulate a grounding error: wrong object or wrong destination."""
        if self.rng.random() >= self.wrong_target_prob:
            return goal
        obj, bin_name, place = goal
        names = [b.name for b in self.env.layout.blocks if b.name != obj]
        bins = [b.name for b in self.env.layout.bins if b.name != bin_name]
        if bin_name is not None and bins and self.rng.random() < 0.5:
            return (obj, str(self.rng.choice(bins)), place)
        if names:
            return (str(self.rng.choice(names)), bin_name, place)
        return goal

    # --------------------------------------------------------------- policy

    def act(self, obs: Observation) -> Decision:
        env = self.env
        held = env.held_objects()
        plan = env.task.plan(env)
        pending = [g for g in plan if not self._goal_done(g)]
        if not pending and not held:
            return Decision([], self.source, "All goals satisfied.", done=True)

        if held:
            obj = held[0]
            goal = self._intent if self._intent and self._intent[0] == obj else next((g for g in plan if g[0] == obj), None)
            if goal is None:  # holding something that has no goal: put it back down nearby
                goal = (obj, None, (*env.eef_xyz()[:2], env.layout_block(obj).half))
            return self._place(obj, goal)

        goal = self._maybe_corrupt(pending[0])
        self._intent = goal
        return self._pick(goal[0])

    def _pick(self, obj: str) -> Decision:
        env = self.env
        p = env.object_pos(obj)
        yaw = self.grasp_yaw_deg(obj)
        v = env.object_vel(obj)
        v[2] = 0.0
        if np.linalg.norm(v) > 0.01:  # moving target: lead it
            travel = np.linalg.norm(env.eef_xyz()[:2] - p[:2]) / 0.25 + 1.2
            p_above = p + v * travel
            p_grasp = p + v * (travel + 0.5)
            xy_above, xy_grasp = p_above[:2], p_grasp[:2]
        else:
            xy_above = xy_grasp = self._noisy(p[:2])
        grasp_z = max(p[2], 0.012)
        wps = [
            Waypoint((*xy_above, max(CARRY_Z, grasp_z + 0.12)), yaw, "open"),
            Waypoint((*xy_grasp, grasp_z), yaw, "close"),
            Waypoint((*xy_grasp, grasp_z + 0.06), yaw, None),
        ]
        return Decision(wps, self.source, f"Pick {obj}.")

    def _place(self, obj: str, goal) -> Decision:
        env = self.env
        _, bin_name, place = goal
        half = env.layout_block(obj).half
        if bin_name is not None:
            b = env.layout_bin(bin_name)
            xy = self._noisy(b.xy)
            release_z = b.height + half + 0.02
        else:
            xy = self._noisy(place[:2])
            release_z = place[2] + 0.004
        yaw = obs_yaw = math.degrees(env.target_yaw)
        if bin_name is None:
            # Align the carried block with whatever it is being stacked on.
            yaw = obs_yaw
        wps = [
            Waypoint((*env.eef_xyz()[:2], CARRY_Z), yaw, None),
            Waypoint((*xy, CARRY_Z), yaw, None),
            Waypoint((*xy, release_z), yaw, "open"),
            Waypoint((*xy, CARRY_Z), yaw, None),
        ]
        return Decision(wps, self.source, f"Place {obj} {'in ' + bin_name if bin_name else 'at ' + str(np.round(place, 3).tolist())}.")
