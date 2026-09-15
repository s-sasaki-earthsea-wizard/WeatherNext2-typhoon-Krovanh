## ---- RunPod ----
RUNPOD_HOST    ?= runpod
RUNPOD_WORKDIR ?= /workspace/WeatherNext2-typhoon-Krovanh

runpod-ssh: ## Open an SSH session to the pod
	ssh $(RUNPOD_HOST)

runpod-push-inputs: ## Upload prepared inputs for CASE to the pod
	RUNPOD_HOST=$(RUNPOD_HOST) RUNPOD_WORKDIR=$(RUNPOD_WORKDIR) bash runpod/sync.sh push $(CASE)

runpod-pull-outputs: ## Download tracks and cropped fields for CASE from the pod
	RUNPOD_HOST=$(RUNPOD_HOST) RUNPOD_WORKDIR=$(RUNPOD_WORKDIR) bash runpod/sync.sh pull $(CASE)
