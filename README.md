# MiniFrame

A lightweight columnar DataFrame engine built from scratch in Python 3.12, featuring a Polars-inspired API.

## Planned Features

- Columnar storage with explicit null handling
- Immutable Series and DataFrame
- Joins, grouping, sorting, and window functions
- Custom binary file format and CSV support
- Comprehensive automated testing

## Documentation

- [Design Document](docs/design-doc.md)

## Status

🚧 In development — infrastructure initialized, core implementation in progress.

## Development

Use uv to set up Python 3.12 and install MiniFrame with its development dependencies.

```bash
uv sync
```

Run the tests, lint checks, formatting checks, and branch coverage report.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=miniframe --cov-branch
```

Hypothesis uses the `dev` profile by default. Set `HYPOTHESIS_PROFILE` to `ci` or
`mutation` to select a different profile.
