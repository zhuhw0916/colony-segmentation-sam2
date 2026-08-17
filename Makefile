.PHONY: test format check

test:
	python -m pytest -q

format:
	black src tests

check:
	python -m compileall -q src tests
	python -m pytest -q
	black --check src tests
