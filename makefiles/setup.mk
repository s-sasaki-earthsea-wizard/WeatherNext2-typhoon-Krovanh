## ---- Environment ----
setup: ## Create the local (Mac) venv: base + analysis + dev groups
	uv sync --group analysis --group dev

setup-gpu: ## Create the RunPod venv: base + gpu group (Linux only)
	uv sync --group gpu

lock: ## Re-resolve uv.lock
	uv lock

test: ## Run unit tests
	uv run pytest -q

lint: ## Run ruff on src, scripts, tests
	uv run ruff check src scripts tests
