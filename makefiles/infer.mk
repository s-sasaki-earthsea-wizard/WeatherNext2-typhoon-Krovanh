## ---- Inference (RunPod H100) ----
smoke-gpu: ## Check that JAX sees the GPU
	uv run python -c "import jax; print(jax.default_backend(), jax.devices())"

infer: ## Run the WN2 ensemble rollout for CASE
	uv run python scripts/run_inference.py --config $(CONFIG) --case $(CASE)

track: ## Run the bundled cyclone tracker on forecast output for CASE
	uv run python scripts/run_tracker.py --config $(CONFIG) --case $(CASE)
