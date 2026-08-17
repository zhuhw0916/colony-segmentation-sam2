.PHONY: test format check

test:
	python -m pytest -q

format:
	black .

check:
	python -m compileall -q .
	python -m pytest -q
	black --check .
