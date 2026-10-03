"""Evaluate an armlab VLM policy on RoboLab tasks (run with Isaac Lab's Python).

  OMNI_KIT_ACCEPT_EULA=Y python -m armlab.robolab.run_llm_eval --policy direct \
      --vlm anthropic:claude-opus-5-5:medium --task BananaInBowlTask --headless --enable-gt-state

Mirrors RoboLab's policies/pi0_family/run.py, but registers the absolute end-effector IK action
space so the model can command end-effector targets (the report's "direct" interface).
"""
import argparse
import json
import os
import sys
import traceback

import cv2  # noqa: F401 -- must be imported before isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", choices=["direct"], default="direct",
                    help="'hybrid' needs a learned System-1 (e.g. pi0.5) server; see README for wiring it.")
parser.add_argument("--vlm", default="anthropic:claude-opus-5-5:medium")
parser.add_argument("--no-overlay", action="store_true")
parser.add_argument("--log-file", default="armlab_robolab_decisions.jsonl")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa: E402

from armlab.policy.llm import DirectVLMPolicy  # noqa: E402
from armlab.policy.vlm import make_vlm  # noqa: E402
from armlab.robolab.client import VLMRoboLabClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
auto_register_droid_abs_ik_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

class StreamingLog(list):
    """Decision log that is also appended to --log-file as it grows, so a long or crashed run can be inspected."""

    def append(self, row):
        super().append(row)
        with open(args_cli.log_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[armlab] decision {len(self)}: {row.get('think_s', 0):.1f}s {str(row.get('rationale', ''))[:160]}",
              flush=True)


open(args_cli.log_file, "w").close()
LOG: list = StreamingLog()


def make_client(args):
    policy = DirectVLMPolicy(vlm=make_vlm(args.vlm), overlay=not args.no_overlay)
    client = VLMRoboLabClient(policy, use_gt_state=args.enable_gt_state, log=LOG)
    orig_reset = client.reset

    def reset(*, env_id=None):
        orig_reset(env_id=env_id)
        policy.reset("")
    client.reset = reset
    return client


def main() -> int:
    # SimulationApp.close() hard-exits the process (exit code 0), so anything raised inside run_evaluation must be
    # printed and turned into an exit code *before* closing, or the run silently "succeeds" with no episodes.
    rc = 0
    try:
        run_evaluation(args_cli, policy=f"armlab-{args_cli.policy}", client_factory=make_client)
    except BaseException as e:  # noqa: BLE001
        print(f"[armlab] terminated with error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        rc = 1
    finally:
        print(f"[armlab] {len(LOG)} decisions logged to {args_cli.log_file}; exit code {rc}", flush=True)
    return rc


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    if code:
        os._exit(code)  # skip simulation_app.close(), which would exit 0
    simulation_app.close()
