"""VLM-in-the-loop policies, mirroring the two architectures in the GPT-6 Astra report.

DirectVLMPolicy  ("GPT 6 Astra (direct)"): the model reads images + proprioception + history and
                 emits 1-5 end-effector waypoints per decision.
HybridPolicy     ("pi0.5 + GPT 6 Astra"): a System-1 policy proposes a waypoint segment; the model
                 accepts a prefix of it or replaces it with its own 1-5 waypoint correction.

Python handles validation, grounding helpers and the simulator connection; the model makes the
action decisions (same split as the report).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Decision, Observation, Policy, Usage, Waypoint
from .vlm import VLM, extract_json

MAX_WAYPOINTS = 5
FONT = None


def _font(size=12):
    global FONT
    if FONT is None:
        try:
            FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            FONT = ImageFont.load_default()
    return FONT


# ------------------------------------------------------------------------------- prompts

SYSTEM_BASE = """You are the controller of a single Franka Panda arm with a parallel-jaw gripper, working at a table in simulation.

COORDINATES. Robot base frame in meters: +x points away from the robot base, +y is the robot's left, +z is up. The table surface is z = 0. Reachable area is roughly x 0.25..0.75, y -0.45..0.45, z 0.01..0.5.

ACTIONS. You command the gripper's fingertip center (the point midway between the finger pads) with waypoints. Each waypoint has a target position, a yaw in degrees about the vertical axis, and a gripper command ("open", "close" or "none") that is applied AFTER the position is reached. At yaw 0 the fingers close along the x axis; at yaw 90 they close along y. The gripper always points straight down. The arm moves in straight lines at about 0.25 m/s. Give 1 to 5 waypoints per decision; after they execute you receive a fresh observation and decide again. Positions outside the reachable area are clamped.

TARGETS. A waypoint target can be:
  - "xyz": absolute [x, y, z] in meters, or
  - "pixel": a pixel (u, v) in the HEAD image (u = column from the left, v = row from the top, in the image's own pixel units). The harness looks up the 3D surface point under that pixel using the depth camera and adds z_offset (meters) to its height. Pointing at the middle of a block's top face with z_offset = -half the block height puts the fingertips at the block's center.
You can also request "locate" queries: a list of head-image pixels whose 3D surface points you want reported. If you request locations and give no waypoints, you get the answers immediately (same observation) and decide again.

USEFUL FACTS. Blocks are cubes, 4 cm unless stated otherwise (top face at z = 0.04, center at z = 0.02 when resting on the table). A 4 cm block is held when the reported gripper width is about 0.04 m after closing; a width near 0 means the gripper closed on nothing. Bins have walls about 6 cm tall: carry objects at z >= 0.15, lower to about z = 0.10-0.12 above the bin center, then open. Approach blocks from above (z about 0.15), descend vertically, close, then lift. Align the gripper yaw with the block's faces.

FEEDBACK. The simulator reports task_done = true as soon as the task's success condition holds, and the episode then ends. If you believe you are finished but task_done is still false, something is not yet right: re-inspect the scene (e.g. an object missed the bin, the order is wrong, an object is still held) and fix it.

Keep short persistent notes about your plan and progress in "notes"; they are shown back to you at the next decision.
"""

DIRECT_TASK = """Respond with JSON only, following the schema: summarize what you observe, explain your plan for the next segment, then give waypoints (or locate queries)."""

HYBRID_TASK = """A learned visuomotor policy (System 1) has proposed the candidate waypoint segment listed below; its waypoints are drawn on the HEAD image as numbered orange dots joined by a line (the current fingertip position is the magenta cross). System 1 moves fluently but sometimes picks the wrong object, the wrong destination or an imprecise position.

Choose one:
  - "accept": execute the first `accept_count` candidate waypoints unchanged (1..all).
  - "correct": ignore the candidate and give your own 1-5 waypoints instead.
Prefer accepting when the candidate serves the task; correct only when it is wrong or unsafe. Respond with JSON only."""


def _waypoint_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "target_type": {"type": "string", "enum": ["xyz", "pixel"]},
            "xyz": {"type": "array", "items": {"type": "number"}},
            "pixel_u": {"type": "integer"},
            "pixel_v": {"type": "integer"},
            "z_offset": {"type": "number"},
            "yaw_deg": {"type": "number"},
            "gripper": {"type": "string", "enum": ["open", "close", "none"]},
        },
        "required": ["target_type", "xyz", "pixel_u", "pixel_v", "z_offset", "yaw_deg", "gripper"],
        "additionalProperties": False,
    }


def decision_schema(hybrid: bool) -> dict:
    props = {
        "observation": {"type": "string"},
        "plan": {"type": "string"},
        "locate": {"type": "array", "items": {
            "type": "object", "properties": {"u": {"type": "integer"}, "v": {"type": "integer"}},
            "required": ["u", "v"], "additionalProperties": False}},
        "waypoints": {"type": "array", "items": _waypoint_schema()},
        "task_complete": {"type": "boolean"},
        "notes": {"type": "string"},
    }
    required = ["observation", "plan", "locate", "waypoints", "task_complete", "notes"]
    if hybrid:
        props["choice"] = {"type": "string", "enum": ["accept", "correct"]}
        props["accept_count"] = {"type": "integer"}
        required += ["choice", "accept_count"]
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


# ------------------------------------------------------------------------------- overlays

GRID_X = np.round(np.arange(0.2, 0.81, 0.1), 2)
GRID_Y = np.round(np.arange(-0.4, 0.41, 0.1), 2)


def annotate_head(obs: Observation, candidate: list[Waypoint] | None = None, grid: bool = True) -> np.ndarray:
    """Draw the table coordinate grid, the current fingertip and (optionally) candidate waypoints."""
    img = Image.fromarray(obs.images["head"]).convert("RGB")
    proj = obs.xyz_to_pixel
    if proj is None:
        return np.asarray(img)
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(11)
    if grid:
        for x in GRID_X:
            a, b = proj("head", (x, GRID_Y[0], 0)), proj("head", (x, GRID_Y[-1], 0))
            if a and b:
                d.line([a, b], fill=(255, 255, 255, 70), width=1)
                d.text((b[0] + 3, b[1] - 7), f"x={x:.1f}", font=f, fill=(255, 255, 255, 200))
        for y in GRID_Y:
            a, b = proj("head", (GRID_X[0], y, 0)), proj("head", (GRID_X[-1], y, 0))
            if a and b:
                d.line([a, b], fill=(255, 255, 255, 70), width=1)
                d.text((b[0] - 14, b[1] - 16), f"y={y:+.1f}", font=f, fill=(255, 255, 255, 200))
    e = proj("head", obs.eef_xyz)
    if e:
        d.line([(e[0] - 9, e[1]), (e[0] + 9, e[1])], fill=(255, 0, 255, 255), width=2)
        d.line([(e[0], e[1] - 9), (e[0], e[1] + 9)], fill=(255, 0, 255, 255), width=2)
    if candidate:
        pts = [proj("head", w.xyz) for w in candidate]
        prev = e
        for i, p in enumerate(pts, 1):
            if p is None:
                continue
            if prev:
                d.line([prev, p], fill=(255, 150, 0, 220), width=2)
            d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(255, 150, 0, 230))
            d.text((p[0] - 3, p[1] - 7), str(i), font=f, fill=(0, 0, 0, 255))
            prev = p
    return np.asarray(img)


# ------------------------------------------------------------------------------- policy core


@dataclass
class HistoryItem:
    index: int
    source: str
    summary: str
    waypoints: list[dict]
    feedback: str


@dataclass
class VLMPolicyBase(Policy):
    vlm: VLM
    history_len: int = 8
    max_locate_rounds: int = 2
    overlay: bool = True
    history: list[HistoryItem] = field(default_factory=list)
    notes: str = ""
    instruction: str = ""
    parse_failures: int = 0

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.history = []
        self.notes = ""
        self.parse_failures = 0

    def record(self, decision: Decision, feedback: str) -> None:
        self.history.append(HistoryItem(len(self.history) + 1, decision.source, decision.rationale,
                                        [w.to_dict() for w in decision.waypoints], feedback))

    def _state_text(self, obs: Observation) -> str:
        lines = [f"TASK: {obs.instruction}",
                 f"task_done: {str(obs.task_done).lower()}",
                 f"time: {obs.sim_time:.1f} s (step {obs.step} of {obs.max_steps})",
                 f"fingertip xyz: {np.round(obs.eef_xyz, 3).tolist()}  yaw_deg: {obs.eef_yaw_deg:.0f}  gripper width: {obs.gripper_width:.3f} m"]
        if obs.objects:
            lines.append("OBJECT STATE (from perception): " + json.dumps(obs.objects))
        if obs.feedback:
            lines.append(f"LAST SEGMENT: {obs.feedback}")
        if self.notes:
            lines.append(f"YOUR NOTES: {self.notes}")
        if self.history:
            lines.append("RECENT DECISIONS:")
            for h in self.history[-self.history_len:]:
                lines.append(f"  #{h.index} [{h.source}] {h.summary} | waypoints {json.dumps(h.waypoints)} | {h.feedback}")
        cams = "; ".join(f"{k.upper()}: {v}" for k, v in obs.camera_info.items())
        lines.append(f"CAMERAS: {cams}. HEAD image size: {obs.images['head'].shape[1]}x{obs.images['head'].shape[0]} px." +
                     (" The head image has the table's x/y grid lines (every 0.1 m) and the fingertip (magenta cross) drawn on it." if self.overlay else ""))
        return "\n".join(lines)

    def _images(self, obs: Observation, candidate=None) -> list:
        head = annotate_head(obs, candidate, grid=self.overlay) if (self.overlay or candidate) else obs.images["head"]
        return ["HEAD camera:", head, "WRIST camera:", obs.images["wrist"]]

    def _to_waypoints(self, obs: Observation, raw: list[dict]) -> list[Waypoint]:
        out = []
        for w in raw[:MAX_WAYPOINTS]:
            g = w.get("gripper")
            g = None if g in (None, "none") else g
            if w.get("target_type") == "pixel":
                if obs.pixel_to_xyz is None:
                    raise ValueError("pixel targets unavailable")
                p = obs.pixel_to_xyz("head", int(w["pixel_u"]), int(w["pixel_v"]))
                if p is None:
                    raise ValueError(f"no surface at pixel {(w['pixel_u'], w['pixel_v'])}")
                xyz = (float(p[0]), float(p[1]), float(p[2]) + float(w.get("z_offset", 0.0)))
            else:
                xyz = tuple(float(v) for v in w["xyz"])
                if len(xyz) != 3:
                    raise ValueError(f"xyz needs 3 numbers: {w['xyz']}")
            out.append(Waypoint(xyz, float(w.get("yaw_deg", 0.0)), g))
        return out

    def _query(self, obs: Observation, task_text: str, hybrid: bool, candidate=None) -> tuple[dict, Usage, list[Waypoint]]:
        usage = Usage()
        extra: list[str] = []
        last_err = None
        for _round in range(self.max_locate_rounds + 2):
            parts = [*self._images(obs, candidate), self._state_text(obs)]
            if candidate is not None:
                parts.append("CANDIDATE (System 1): " + json.dumps([w.to_dict() for w in candidate]))
            parts += extra
            parts.append(task_text)
            text, u = self.vlm.complete(SYSTEM_BASE, parts, decision_schema(hybrid))
            usage.add(u)
            try:
                d = extract_json(text)
                if d.get("locate") and not d.get("waypoints") and _round < self.max_locate_rounds and not (hybrid and d.get("choice") == "accept"):
                    answers = []
                    for q in d["locate"][:12]:
                        p = obs.pixel_to_xyz("head", int(q["u"]), int(q["v"])) if obs.pixel_to_xyz else None
                        answers.append({"u": q["u"], "v": q["v"], "xyz": None if p is None else np.round(p, 3).tolist()})
                    extra = [f"LOCATE RESULTS (head image pixel -> 3D surface point): {json.dumps(answers)}. Now give your decision."]
                    continue
                wps = self._to_waypoints(obs, d.get("waypoints") or [])
                return d, usage, wps
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                last_err = e
                self.parse_failures += 1
                extra = [f"Your previous reply could not be used ({e}). Reply again with valid JSON per the schema."]
        raise RuntimeError(f"model did not return a usable decision: {last_err}")


class DirectVLMPolicy(VLMPolicyBase):
    name = "direct"

    def act(self, obs: Observation) -> Decision:
        d, usage, wps = self._query(obs, DIRECT_TASK, hybrid=False)
        self.notes = d.get("notes", self.notes)
        rationale = d.get("plan", "")
        return Decision(wps, "llm", rationale, done=bool(d.get("task_complete")) and not wps, usage=usage,
                        notes=self.notes)


@dataclass
class HybridPolicy(VLMPolicyBase):
    system1: Policy | None = None
    name: str = "hybrid"

    def reset(self, instruction: str) -> None:
        super().reset(instruction)
        if self.system1 is not None:
            self.system1.reset(instruction)

    def act(self, obs: Observation) -> Decision:
        proposal = self.system1.act(obs)
        candidate = proposal.waypoints
        if not candidate:
            # System 1 has nothing to propose (believes it is done): let the model drive this step.
            d, usage, wps = self._query(obs, DIRECT_TASK, hybrid=False)
            self.notes = d.get("notes", self.notes)
            return Decision(wps, "llm-correction", d.get("plan", ""), done=bool(d.get("task_complete")) and not wps,
                            usage=usage, candidate=[], notes=self.notes)
        d, usage, wps = self._query(obs, HYBRID_TASK, hybrid=True, candidate=candidate)
        self.notes = d.get("notes", self.notes)
        if d.get("choice") == "accept" or not wps:
            k = int(d.get("accept_count") or len(candidate))
            k = max(1, min(k, len(candidate)))
            return Decision(candidate[:k], "system1", d.get("plan", "") or proposal.rationale, usage=usage,
                            candidate=candidate, notes=self.notes)
        return Decision(wps, "llm-correction", d.get("plan", ""), usage=usage, candidate=candidate, notes=self.notes)
