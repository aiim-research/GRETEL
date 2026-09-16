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

