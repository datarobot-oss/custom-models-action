
FUNCTIONAL_TESTS ?= 'tests/functional'

validate-env-%:
	@if [ "${$*}" = "" ]; then \
	  printf "\033[0;31m\nEnvironment variable '${*}' is not set. Aborting.\n\n\033[0m"; \
	  exit 1; \
	fi

reqs:
	pip install -r tests/requirements.txt
.PHONY: reqs

test: test-unit
.PHONY: test

test-full: test-unit test-functional
.PHONY: test-full

test-unit:
	set -ex; PYTHONPATH=.:src pytest tests/unit
.PHONY: test-unit

test-functional: validate-env-DATAROBOT_WEBSERVER validate-env-DATAROBOT_API_TOKEN
	set -ex; PYTHONPATH=.:src pytest -v --log-cli-level error ${FLAGS} ${FUNCTIONAL_TESTS}

.PHONY: test

black:
	@isort --profile black .
	@black .
.PHONY: black

black-check:
	@black --check .

.PHONY: black-check

lint: black-check
	@flake8 .
	@isort --check-only .
	@pylint --recursive=y .
.PHONY: lint
