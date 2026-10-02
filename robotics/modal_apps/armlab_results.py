"""Phone-friendly results page over the `armlab-runs` Modal volume: runs, summary tables, playable videos,
and the daily safety reports. Holds no model keys.

  modal deploy modal_apps/armlab_results.py        # prints https://<workspace>--armlab-results-web.modal.run

The URL is public but unguessable. To require a key, deploy with ARMLAB_RESULTS_KEY set in your shell and open
the page once as `/?key=<that key>` (a cookie remembers the browser):

  ARMLAB_RESULTS_KEY=$(openssl rand -hex 12) modal deploy modal_apps/armlab_results.py
"""
import os

import modal

# Pinned bookworm base + explicit minimums: older workspaces' debian_slim ships a pydantic-v1-era FastAPI whose
# Starlette cannot answer HTTP Range requests (iOS Safari will not play video without them).
image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .uv_pip_install("fastapi[standard]>=0.115", "starlette>=0.46", "pydantic>=2")
    .add_local_python_source("armlab")
)
app = modal.App("armlab-results", image=image)
runs = modal.Volume.from_name("armlab-runs", create_if_missing=True)
_key = os.environ.get("ARMLAB_RESULTS_KEY")


@app.function(volumes={"/runs": runs}, scaledown_window=300,
              secrets=[modal.Secret.from_dict({"ARMLAB_RESULTS_KEY": _key})] if _key else [])
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import time

    from armlab.web.results import make_app

    last = [0.0]

    def refresh():  # pick up new runs; Volume.reload is cheap but no need to do it on every request
        if time.time() - last[0] > 10:
            runs.reload()
            last[0] = time.time()

    return make_app("/runs", access_key=os.environ.get("ARMLAB_RESULTS_KEY") or None, refresh=refresh)
