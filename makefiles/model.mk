## ---- Model contract and local checks (Mac, CPU) ----
model-spec: ## Print the input contract read from the checkpoint task config
	uv run python scripts/show_model_spec.py --config $(CONFIG)

smoke-mini: ## Run the full pipeline end to end with the 1 deg Mini checkpoint
	uv run python scripts/smoke_mini.py --config $(CONFIG)
