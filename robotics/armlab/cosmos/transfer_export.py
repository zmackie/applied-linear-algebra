"""Sim-to-real data generator: export a sim episode as Cosmos-Transfer2.5 inputs.

Renders RGB, depth and instance-segmentation videos of the same rollout (720p, 16 fps) and writes
one Transfer spec per appearance prompt. Run the specs on a GPU with
`modal run modal_apps/cosmos_transfer.py --export-dir <dir>` to get photoreal variations.

  python -m armlab.cosmos.transfer_export --task blocks_into_bin --seed 0 --out exports/blocks0 --variations 4
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import mujoco
import numpy as np

from ..eval.runner import make_policy
from ..sim.env import CONTROL_DT, ArmEnv
from .reason import write_video

SIZE = (720, 1280)
FPS = 16
DEPTH_NEAR, DEPTH_FAR = 0.8, 3.0

PROMPTS = [
    "A real video of a white Franka robot arm on a light wooden table in a bright, modern robotics lab. "
    "The arm picks up small painted wooden blocks and drops them into a gray plastic storage bin. Soft daylight from large windows, realistic shadows, sharp focus.",
    "A real video recorded in a busy warehouse at night under sodium lights. An industrial robot arm on a scuffed steel workbench sorts small colored blocks into a plastic tote. "
    "Concrete floor and shelving in the background, slight motion blur.",
    "A real video in a cozy home kitchen. A robot arm on a butcher-block countertop picks up colorful toy blocks and puts them into a container. "
    "Warm evening light, houseplants and cabinets in the background.",
    "A real video in a cleanroom with bright, even overhead LED panels. A robot arm on a white laminate table moves small colored cubes into a gray bin. "
    "Clinical, high-key lighting, crisp detail.",
    "A real outdoor video on a patio table on an overcast day. A robot arm picks up wooden blocks and places them into a weathered plastic bin. Diffuse natural light, garden in the background.",
]


def _palette(n: int) -> np.ndarray:
    cols = [colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.75, 0.95) for i in range(n)]
    return (np.array(cols) * 255).astype(np.uint8)


def render_controls(env: ArmEnv, camera: str = "front") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = SIZE
    rgb = env.render(camera, h, w, shadows=True)
    depth = env.render(camera, h, w, depth=True)
    seg = env.render(camera, h, w, segmentation=True)
    inv = 1.0 / np.clip(depth, DEPTH_NEAR, DEPTH_FAR)
    d8 = ((inv - 1 / DEPTH_FAR) / (1 / DEPTH_NEAR - 1 / DEPTH_FAR) * 255).astype(np.uint8)
    depth_rgb = np.repeat(d8[..., None], 3, axis=2)
    m = env.model
    pal = _palette(m.nbody + 1)
    objid, objtype = seg[..., 0], seg[..., 1]
    body = np.full(objid.shape, -1)
    is_geom = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & (objid >= 0)
    body[is_geom] = m.geom_bodyid[objid[is_geom]]
    seg_rgb = np.zeros((h, w, 3), np.uint8)
    seg_rgb[body >= 0] = pal[body[body >= 0]]
    return rgb, depth_rgb, seg_rgb


def export(task: str, seed: int, out: Path, variations: int = 3, seconds: float = 5.8, policy: str = "oracle") -> list[Path]:
    """Run `policy` on the task and export ~`seconds` of control videos plus Transfer specs."""
    out.mkdir(parents=True, exist_ok=True)
    env = ArmEnv(task, seed=seed)
    pol = make_policy(policy, env, None, seed)
    pol.reset(env.task.instruction)
    rgb, depth, seg = [], [], []
    next_t = [0.0]
    n_frames = int(round(seconds * FPS))

    def grab(e, fi):
        if len(rgb) < n_frames and fi.sim_time + 1e-9 >= next_t[0]:
            r, d, s = render_controls(e)
            rgb.append(r)
            depth.append(d)
            seg.append(s)
            next_t[0] += 1.0 / FPS

    env.frame_callbacks.append(grab)
    obs = env.observe()
    while len(rgb) < n_frames and env.budget_left() > 0 and not obs.task_done:
        dec = pol.act(obs)
        if not dec.waypoints:
            env.idle(CONTROL_DT * 5)
        else:
            env.execute(dec.waypoints)
        obs = env.observe()
    while len(rgb) < n_frames and env.budget_left() > 0:
        env.idle(CONTROL_DT * 5)
    env.close()

    write_video(rgb, out / "input_rgb.mp4", FPS)
    write_video(depth, out / "control_depth.mp4", FPS)
    write_video(seg, out / "control_seg.mp4", FPS)
    specs = []
    for i, prompt in enumerate(PROMPTS[:variations]):
        spec = {
            "name": f"{task}_s{seed}_v{i}",
            "prompt": prompt,
            "video_path": "input_rgb.mp4",
            "guidance": 3,
            # Edge and blur controls are computed on the fly from input_rgb.mp4 by Cosmos-Transfer2.5.
            "edge": {"control_weight": 0.5},
            "depth": {"control_path": "control_depth.mp4", "control_weight": 0.5},
            "seg": {"control_path": "control_seg.mp4", "control_weight": 0.5},
        }
        p = out / f"spec_v{i}.json"
        p.write_text(json.dumps(spec, indent=2))
        specs.append(p)
    (out / "export.json").write_text(json.dumps({"task": task, "seed": seed, "frames": len(rgb), "fps": FPS,
                                                 "size": list(SIZE), "specs": [s.name for s in specs]}, indent=2))
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="blocks_into_bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variations", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=5.8, help="93 frames at 16 fps = one Transfer chunk")
    args = ap.parse_args(argv)
    out = Path(args.out or f"exports/{args.task}_s{args.seed}")
    for s in export(args.task, args.seed, out, args.variations, args.seconds):
        print(s)


if __name__ == "__main__":
    main()
