## ---- Analysis (Mac) ----
evaluate: ## Compare forecast tracks with the JMA best track for CASE
	uv run python scripts/evaluate.py --config $(CONFIG) --case $(CASE)
