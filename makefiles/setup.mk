## ---- Environment ----
setup: ## Create the local (Mac) venv: base + model (CPU jax) + analysis + dev
	uv sync --group cpu --group analysis --group dev

setup-gpu: ## Create the RunPod venv: base + model + CUDA jax (Linux only)
	uv sync --group gpu

lock: ## Re-resolve uv.lock
	uv lock

test: ## Run unit tests
	uv run pytest -q

lint: ## Run ruff on src, scripts, tests
	uv run ruff check src scripts tests
