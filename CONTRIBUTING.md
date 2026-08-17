# Contributing

## Development setup

Use Python 3.10 or newer. For the benchmark environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

SAM2 itself and its checkpoints are external dependencies. Follow the SAM2
project's installation and model-download instructions separately.

## Before opening a pull request

Run the same checks used by continuous integration:

```bash
python -m compileall -q .
python -m pytest -q
black --check .
```

Keep datasets, model checkpoints, generated masks, benchmark CSVs, logs, and
machine-specific paths out of commits. Add or update tests when changing
preprocessing, mask filtering, or output formats.

## Pull requests

Describe the motivation, the files changed, the command used to test the
change, and any GPU/model requirements. Keep unrelated refactors out of the
same pull request.
