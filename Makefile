
test:
	set -ex; PYTHONPATH=src pytest tests/unit
.PHONY: test


black:
	@black .
.PHONY: black


hello:
	@echo Hello
.PHONY: hello
