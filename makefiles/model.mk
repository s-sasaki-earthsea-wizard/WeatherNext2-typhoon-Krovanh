## ---- Model contract and local checks (Mac, CPU) ----
model-spec: ## Print the input contract read from the checkpoint task config
	uv run python scripts/show_model_spec.py --config $(CONFIG)
