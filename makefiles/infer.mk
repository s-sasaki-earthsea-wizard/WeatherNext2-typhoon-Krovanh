## ---- Inference (RunPod H100) ----
smoke-gpu: ## Check that JAX sees the GPU
	uv run python -c "import jax; print(jax.default_backend(), jax.devices())"

infer: ## Run the WN2 ensemble rollout for CASE
	uv run python scripts/run_inference.py --config $(CONFIG) --case $(CASE)

infer-one: ## Run a single member for CASE, to measure wall-clock before the full run
	uv run python scripts/run_inference.py --config $(CONFIG) --case $(CASE) --members 1

track: ## Re-run the tracker on stored members for CASE (regional crop only)
	uv run python scripts/run_tracker.py --config $(CONFIG) --case $(CASE)
