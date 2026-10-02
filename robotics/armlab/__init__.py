"""armlab: a small testbed for putting vision-language models in a robot-arm control loop."""
import os

# Headless rendering: OSMesa works on CPU-only machines; set MUJOCO_GL=egl on GPU boxes.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
