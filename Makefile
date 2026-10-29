# clean        Remove (!) all untracked and ignored files.
.PHONY: clean
clean:
	git clean -xdff

# ---------------------------------------------------------------------------
# The following commands must be run OUTSIDE the development environment.
# ---------------------------------------------------------------------------

.PHONY: requirements
requirements:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

# docker       Builds the development image from scratch.
.PHONY: docker
docker:
	docker build -t gretelxai/gretel:dev-latest .

# pull         Pull the development image.
.PHONY: pull
pull:
	docker pull gretelxai/gretel:dev-latest

# push         Push the development image to Docker Hub.
.PHONY: push
push:
	docker push gretelxai/gretel:dev-latest

# shell        Opens a shell in the development image.
.PHONY: shell
shell:
	docker-compose run gretel bash

# demo         Run the demo in the development image.
.PHONY: demo
demo:
	docker-compose run gretel

# ---------------------------------------------------------------------------
# The following commands must be run INSIDE the development environment.
# ---------------------------------------------------------------------------

.PHONY: ensure-dev
ensure-dev:
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

# format       Format all source code inplace using `black`.
.PHONY: format
format: ensure-dev
	(git status | grep "nothing to commit") && sudo black autogoal/ tests/ || echo "(!) REFUSING TO REFORMAT WITH UNCOMMITED CHANGES" && exit
	git status

ENV_NAME=GRETEL

env:
	conda activate $(ENV_NAME) && pip-compile requirements.in && pip install -r requirements.txt

kernel:
	conda activate $(ENV_NAME) && python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python 3.12 ($(ENV_NAME))"


# server-import  Bring results, logs and the caches built on the Slurm server (RSGG GANs, BZR and ENZYMES
#                datasets and oracles, DCM medoids) back to this clone, read-only on the server
#                (docs/rerun/README.md, section 4). Only caches written since SERVER_SINCE and missing here
#                come back: the server still holds caches of its own from before the rerun, and a committed
#                cache is never overwritten. Commits nothing: commit only once nothing of the batch is running.
SERVER=kalifano
SERVER_REPO=projects/GRETEL
SERVER_SINCE=2026-10-29
PYTHON=$(shell command -v python || command -v python3)

.PHONY: server-import
server-import:
	ssh -O check $(SERVER) 2>/dev/null || ssh -fN $(SERVER)
	rsync -av $(SERVER):$(SERVER_REPO)/lab/output/results/       lab/output/results/
	rsync -av $(SERVER):$(SERVER_REPO)/lab/output/selector_logs/ lab/output/selector_logs/
	rsync -av $(SERVER):$(SERVER_REPO)/output/logs/              lab/output/server_logs/
	rsync -av $(SERVER):$(SERVER_REPO)/output/slurm/             lab/output/server_slurm/
	@if running=$$(ssh $(SERVER) 'squeue -h -u $$USER -n warm') && [ -z "$$running" ]; then \
	    ssh $(SERVER) "cd $(SERVER_REPO) && find data -name 'GAN-*' -newermt $(SERVER_SINCE); \
	        find lab/data/cache/datasets lab/data/cache/oracles lab/data/cache/explainers -type f \
	             -newermt $(SERVER_SINCE) ! -name '*.lck*' \( -path '*/explainers/*' -name 'DCM-*' -o ! -path '*/explainers/*' \)" \
	        > lab/output/server_slurm/new_caches.txt && \
	    echo "new caches on the server: $$(wc -l < lab/output/server_slurm/new_caches.txt)" && \
	    rsync -av --ignore-existing --files-from=lab/output/server_slurm/new_caches.txt $(SERVER):$(SERVER_REPO)/ ./; \
	else echo "caches left out: a warm job is still running (or squeue could not be reached)"; fi
	$(PYTHON) scripts/collect_logs.py lab/output/server_logs
	git status --short | head -40
