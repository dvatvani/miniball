# Repository Guidelines

## Project Structure & Module Organization
This is a Python package laid out with a `src/`-based import path and supporting tooling.
- `src/miniball/` contains the package code (for example `src/miniball/main.py`).
- `tests/` holds the test suite collected by pytest.
- `playground/` contains Marimo notebooks.
- `pyproject.toml` and `justfile` define tooling, dependencies, and tasks.

## Build, Test, and Development Commands
Use `uv` for dependency management and to run tools. Run tasks with `just <recipe>`.
- `just install` installs dev dependencies and pre-commit hooks.
- `just sync` synchronizes dependencies.
- `just lint` runs Ruff format check and linting.
- `just type-check` runs `ty` for type checking.
- `just tests` runs `uv run pytest -v` and cleans `__pycache__` files.
- `just build` builds the project.
- `uv run marimo edit` opens the Marimo notebook editor in `playground/`.

## Coding Style & Naming Conventions
Follow standard Python conventions.
- Indentation is 4 spaces, no tabs.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Prefer type hints on public APIs; keep docstrings up to date.
- Run `just lint` before opening a pull request.

## Testing Guidelines
Pytest is configured in `pyproject.toml` with coverage for `src/` and JUnit/coverage outputs.
- Place tests in `tests/` and name files `test_*.py`.
- Both pytest and unittest styles are acceptable (pytest will collect both).
- Run tests with `just tests`.

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so no existing commit convention can be summarized here.
- If you use Commitizen, run `cz` to format commits consistently.
- Pull requests should include a concise summary, test results, and any doc updates.
- Add screenshots for documentation or UI-facing changes when relevant.

## Documentation & Configuration Notes
- Python `>=3.13` is required; `uv` `>=0.8.0` is expected.
