## ---- Data (Mac) ----
download-era5: ## Download ERA5 frames for CASE (needs ~/.cdsapirc)
	uv run python scripts/download_era5.py --config $(CONFIG) --case $(CASE)

download-era5-all: ## Download ERA5 frames for every case (shared frame cache)
	uv run python scripts/download_era5.py --config $(CONFIG) --all-cases

prepare-inputs: ## Build the WeatherNext 2 input NetCDF for CASE
	uv run python scripts/prepare_inputs.py --config $(CONFIG) --case $(CASE)

fetch-besttrack: ## Download the JMA reference track (post-analysis if out, else preliminary)
	uv run python scripts/fetch_besttrack.py --config $(CONFIG)
