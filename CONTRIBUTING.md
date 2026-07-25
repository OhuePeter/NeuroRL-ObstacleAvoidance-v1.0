# Contributing

Thank you for your interest in improving this repository.

## Recommended workflow

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Add the upstream remote:

```bash
git remote add upstream https://github.com/OhuePeter/NeuroRL-ObstacleAvoidance-v1.0.git
git fetch upstream
```

4. Create a feature branch from the target branch:

```bash
git checkout -b <feature-name>
```

5. Run formatting, tests, and validation before opening a pull request.
6. Push your branch and open a pull request with a clear description.

## Code standards

- Python 3.11+
- PEP 8 style
- Type hints for new or modified public functions
- Docstrings for non-trivial modules, classes, and functions
- Keep changes focused and avoid unrelated refactors

## Testing expectations

Run:

```bash
pytest -q
```

For analysis or figure-generation changes, include:

- The exact command sequence used.
- Output paths verified.
- Any expected non-determinism or randomness controls.

## Documentation expectations

If your change impacts reproducibility or manuscript assets, update relevant docs in `docs/` and `README.md`.

## Commit and PR quality

- Use descriptive commit messages.
- Keep pull requests reviewable in size.
- Add rationale for scientific or methodological changes.