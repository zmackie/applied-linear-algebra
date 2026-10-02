# applied-linear-algebra

Applied linear algebra, written as [marimo](https://marimo.io) notebooks, plus `robotics/`: a simulated robot-arm testbed for putting vision-language models in the control loop, and Cosmos video-reasoning apps.

## Setup

```sh
uv sync                                        # notebooks
uv pip install -e "robotics[llm,api,cloud,dev]" # robotics testbed (see robotics/README.md)
```

## Usage

```sh
uv run marimo edit notebooks/intro.py        # linear transformations of the unit square
uv run marimo edit notebooks/robot_eval.py   # browse robot-arm eval runs and videos
```

Notebooks live in `notebooks/` as plain `.py` files. The robotics project is documented in [robotics/README.md](robotics/README.md).
