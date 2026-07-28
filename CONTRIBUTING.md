# Contributing

This started as a personal portfolio project, but issues and pull requests
are welcome.

## Setup

```bash
git clone https://github.com/rohithsure2000/pneumonia-xray-cnn.git
cd pneumonia-xray-cnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Before opening a PR

```bash
black src tests
isort src tests
flake8 src tests
pytest
```

## Guidelines

- Keep new model architectures behind `MODEL_REGISTRY` in `models.py` so
  they're automatically picked up by the CLI scripts and covered by the
  shape tests in `tests/test_models.py`.
- Prefer adding a unit test over a manual notebook check -- the point of
  this repo is that results are reproducible without re-running a notebook
  by hand.
- Keep functions importable without TensorFlow installed where possible
  (see the lazy imports in `data.py`/`train.py`); it keeps `--help` and
  config-only tests fast.
