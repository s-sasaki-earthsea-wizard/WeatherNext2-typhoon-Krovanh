# WeatherNext2-typhoon-Krovanh
# Run `make help` for the list of targets. Targets are defined in makefiles/*.mk.
.DEFAULT_GOAL := help

# Local machine settings: ssh alias, pod workdir, results root. The file is
# git-ignored; .env.example lists what belongs in it. It is read before
# makefiles/*.mk, whose defaults are all `?=` and so keep whatever .env set.
#
# make parses this file, it does not source it: plain KEY=value lines only, no
# quotes, no `export`, no command substitution.
-include .env

CONFIG ?= configs/krovanh.yaml
CASE   ?= init-2026-08-31T18

include makefiles/*.mk

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
