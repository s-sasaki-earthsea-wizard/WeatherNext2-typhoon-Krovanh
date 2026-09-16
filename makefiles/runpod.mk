## ---- RunPod ----
RUNPOD_HOST    ?= runpod
RUNPOD_WORKDIR ?= /workspace/WeatherNext2-typhoon-Krovanh
RESULTS_ROOT   ?= /Volumes/EW-NAS-Atoll/Projects/personal-dev/WeatherNext2-typhoon-Krovanh

runpod-ssh: ## Open an SSH session to the pod
	ssh $(RUNPOD_HOST)

runpod-push-inputs: ## Upload prepared inputs for CASE to the pod
	RUNPOD_HOST=$(RUNPOD_HOST) RUNPOD_WORKDIR=$(RUNPOD_WORKDIR) bash runpod/sync.sh push $(CASE)

runpod-pull-outputs: ## Download CASE off the pod and on to the NAS (both legs)
	RUNPOD_HOST=$(RUNPOD_HOST) RUNPOD_WORKDIR=$(RUNPOD_WORKDIR) \
	RESULTS_ROOT=$(RESULTS_ROOT) bash runpod/sync.sh pull $(CASE)

runpod-pull-stage: ## Download CASE to staging/ only, so the pod can be stopped
	RUNPOD_HOST=$(RUNPOD_HOST) RUNPOD_WORKDIR=$(RUNPOD_WORKDIR) bash runpod/sync.sh pull-stage $(CASE)

runpod-pull-publish: ## Copy staged CASE to the NAS; needs no pod, safe to retry
	RESULTS_ROOT=$(RESULTS_ROOT) bash runpod/sync.sh pull-publish $(CASE)
