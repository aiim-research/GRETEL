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

# env          Setup the development environment.
.PHONY: env
env: ensure-dev
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
	ln -s ${HOME}/.poetry/bin/poetry /usr/bin/poetry
	poetry config virtualenvs.create false

# install      Install all the development dependencies.
.PHONY: install
install: ensure-dev
	poetry install
