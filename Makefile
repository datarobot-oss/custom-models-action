
test:
	set -ex; PYTHONPATH=src pytest tests/unit
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