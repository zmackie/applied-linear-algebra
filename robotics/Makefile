# Run from robotics/. Uses the active venv (or ../.venv) if present.
PY ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; else echo python; fi)

.PHONY: doctor test results-local deploy eval-oracle

doctor:            ## which credentials/hosts this machine can reach (never prints secret values)
	$(PY) -m armlab.doctor

test:
	$(PY) -m pytest -q

results-local:     ## preview the results page over ./runs at http://localhost:8000
	$(PY) -m uvicorn --factory armlab.web.results:app_from_env --port 8000

deploy:            ## results page + daily safety cron (Cosmos endpoint: modal deploy modal_apps/cosmos_reason_vllm.py)
	$(PY) -m modal deploy modal_apps/armlab_results.py
	$(PY) -m modal deploy modal_apps/armlab_safety_cron.py

eval-oracle:       ## smoke test on Modal without any model keys
	$(PY) -m modal run modal_apps/armlab_eval.py --policy oracle --tasks all --seeds 0 --video
