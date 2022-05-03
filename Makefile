
test:
	set -ex; PYTHONPATH=src pytest tests/unit
.PHONY: test
