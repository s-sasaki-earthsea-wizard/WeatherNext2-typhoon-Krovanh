## ---- Model contract and local checks (Mac, CPU) ----
model-spec: ## Print the input contract read from the checkpoint task config
	uv run python scripts/show_model_spec.py --config $(CONFIG)

smoke-cpu: ## Run the full pipeline locally on the CPU with the 1 deg Mini model
	uv run python scripts/smoke_cpu.py --config $(CONFIG)
