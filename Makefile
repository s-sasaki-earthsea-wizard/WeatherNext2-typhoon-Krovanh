# WeatherNext2-typhoon-Krovanh
# Run `make help` for the list of targets. Targets are defined in makefiles/*.mk.
.DEFAULT_GOAL := help

CONFIG ?= configs/krovanh.yaml
CASE   ?= init-2026-08-31T18

include makefiles/*.mk

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
