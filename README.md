# miniball

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Quickstart

### 1. Install `uv`

This project uses [uv](https://github.com/astral-sh/uv) as package manager.
Installation instructions are [here](https://docs.astral.sh/uv/getting-started/installation/).
It's recommended to install the latest version from [github releases](https://github.com/astral-sh/uv/releases).


Linux/MacOS (bash):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (powershell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you have already installed `uv`, please ensure you're using the latest version by running `uv self update`.

### 2. Clone repository

```bash
git clone https://github.com/dvatvani/miniball
cd miniball
```


### 4. Run initial setup
   ```bash
   just install
   git add .
   git commit -m "feat: first commit"
   ```

Congratulations! You're now ready to use your new project :sparkles:

## Justfile commands

Run tasks with `just <recipe>`. All commands are defined in-place in `justfile`.

- `build` — Remove `dist/` and build the project with `uv build`.
- `install` — Run `sync` and install pre-commit hooks.
- `lint` — Check code format and lint with [ruff](https://github.com/astral-sh/ruff).
- `github-tag` — Bump version with Commitizen, create a git tag, and push.
- `print-version` — Print the current project version.
- `sync` — Synchronize dependencies with `uv sync`.
- `tests` — Run pytest and remove `__pycache__`/`.pyc` files.
- `type-check` — Run type checking with [ty](https://github.com/astral-sh/ty).

## Marimo notebooks

The Marimo notebooks can be run locally using the following command:

```bash
uv run marimo edit
```

This opens the Marimo editor for interactive notebook development.
