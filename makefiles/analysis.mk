## ---- Analysis (Mac) ----
# Map background for the track figures: natural-earth (default, works offline
# once cartopy has cached its shapefiles) or osm (OpenStreetMap tiles; needs
# the network on the first draw, cached under data/cache/tiles/ after that).
BASEMAP ?= natural-earth

evaluate: ## Compare forecast tracks with the JMA best track for CASE (BASEMAP=osm for map tiles)
	uv run python scripts/evaluate.py --config $(CONFIG) --case $(CASE) --basemap $(BASEMAP)

evaluate-all: ## Run the comparison for every case that has forecast output
	uv run python scripts/evaluate.py --config $(CONFIG) --all-cases --basemap $(BASEMAP)

evaluate-tables: ## Comparison tables only for CASE, skipping the figures
	uv run python scripts/evaluate.py --config $(CONFIG) --case $(CASE) --no-figures
