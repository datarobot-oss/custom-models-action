
validate-env-%:
	@if [ "${$*}" = "" ]; then \
	  printf "\033[0;31m\nEnvironment variable '${*}' is not set. Aborting.\n\n\033[0m"; \
	  exit 1; \
	fi

test: test-unit
.PHONY: test

test-full: test-unit test-functional
.PHONY: test-full

test-unit:
	set -ex; PYTHONPATH=.:src pytest tests/unit
.PHONY: test-unit

test-functional: validate-env-DATAROBOT_WEBSERVER validate-env-DATAROBOT_API_TOKEN
	set -ex; PYTHONPATH=.:src pytest tests/functional
.PHONY: test

black:
	@black .
.PHONY: black

black-check:
	@black --check .
.PHONY: black-check

lint: black-check
	@pycodestyle .
.PHONY: lint
