# Project tasks (in-situ commands; see https://just.systems/)

set shell := ["bash", "-euc"]

[private]
default:
  @just --list

# Synchronize dependencies from lockfile
sync:
    uv sync

# Install dev dependencies and pre-commit hooks
install: sync
    uv run pre-commit install

# Run tests and remove __pycache__ / .pyc
test: sync
    uv run pytest -v
    find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# Run code quality tools.
check:
	@echo "Checking uv lock file consistency with 'pyproject.toml'"
	@uv lock --check
	@echo "Linting, formatting and type-checking code"
	@uv run pre-commit run -a
