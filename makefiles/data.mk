## ---- Data (Mac) ----
download-era5: ## Download ERA5 frames for CASE (needs ~/.cdsapirc)
	uv run python scripts/download_era5.py --config $(CONFIG) --case $(CASE)

prepare-inputs: ## Build the WeatherNext 2 input NetCDF for CASE
	uv run python scripts/prepare_inputs.py --config $(CONFIG) --case $(CASE)

fetch-besttrack: ## Download the JMA position table CSV (official post-analysis)
	uv run python scripts/fetch_besttrack.py --config $(CONFIG) --case $(CASE)
