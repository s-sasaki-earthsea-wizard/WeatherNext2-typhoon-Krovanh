## ---- Analysis (Mac) ----
evaluate: ## Compare forecast tracks with the JMA best track for CASE
	uv run python scripts/evaluate.py --config $(CONFIG) --case $(CASE)

evaluate-all: ## Run the comparison for every case that has forecast output
	uv run python scripts/evaluate.py --config $(CONFIG) --all-cases

evaluate-tables: ## Comparison tables only for CASE, skipping the figures
	uv run python scripts/evaluate.py --config $(CONFIG) --case $(CASE) --no-figures
