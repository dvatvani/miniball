# Miniball

**2D five-a-side football in Python** — simple rules, headless simulation, and a clean API so you can pit your own bots against bundled baselines or a full round-robin league.

Play with a controller or watch AIs play; the key point in this project is to model strategy, not to simulate real-world physics.

### Gameplay / Rules

- Five-a-side, top-down 2D pitch.
- No inertia, sprint, or stamina: each player either stands still or moves at a fixed speed (stateless movement).
- Ball sticks to a player on contact; defenders can tackle by colliding with the carrier.
- One shot type: strike instantly along your current move direction at fixed speed — passes, clears, and shots are the same mechanic.
- Controller play has no pass/shot assist (fully manual aim).
- Ball touch cooldown (~1s) after losing or striking the ball to limit exploits.
- No offsides, fouls, or set pieces; walled pitch keeps the ball in play.
- Kickoffs: furthest-forward home player starts with the ball; after a goal, the conceding team restarts the same way.
- Players cannot overlap; pushing resolves contested positions.

---

## Requirements

<details>

<summary>uv</summary>

### Install `uv`

This project uses [uv](https://github.com/astral-sh/uv) as package manager.
Installation instructions are [here](https://docs.astral.sh/uv/getting-started/installation/).

</details>

<details>

<summary>just (Optional, but recommended for a nicer CLI experience)</summary>

### Install `just`

This project uses [just](https://just.systems/man/en/) as an alternative to `makefile` to provide easy CLI access to common CLI operations in this project.
Installation instructions are [here](https://github.com/casey/just?tab=readme-ov-file#installation).

</details>

<br>

If you prefer not to install `just`, then whenever the instructions below include commands like `just <X>`, look inside the `justfile` file in the project and run the commands listed under the relevant just command recipe manually instead.

## Setup

### 1. Clone repository

```bash
git clone https://github.com/dvatvani/miniball
cd miniball
```

### 2. Run install command
```bash
just install
```

Congratulations! You're now ready to start playing :sparkles:

## Play

### Playing a game through the UI
```bash
just play
```

Having a joystick / controller is recommended if you'd like to take control of a player.

### Fast AI vs AI match simulations

Headless AI vs AI match simulations can be run from the CLI
```bash
uv run -m miniball.simulation
```

Running the above with a `--help` flag lists the CLI options available.

The results of the match simulation are stored in the `match_data` directory, and can be analysed with the `notebooks/analyse_saved_match.py` marimo notebook (`uv run marimo run notebooks/analyse_saved_match.py`).

Matches can also be simulated easily from within Python:

```python
from miniball.teams import teams_list
from miniball.simulation import simulate_matches
results = simulate_matches([(teams_list[0], teams_list[1])])
```

### League simulations

Round-robin league simulations including all AI models can be run from the CLI
```bash
just run-league-simulation
```

League simulations can also be easily run from within Python:

```python
from miniball.league import simulate_league
league_table_df = simulate_league()  # teams to include can be passed in. Uses all by default
```

## Developing your own bots

1. **Subclass `BaseAI`** (`miniball.ai.BaseAI`). Implement `get_actions(self, state: GameState) -> TeamActions`: a mapping from shirt number to `{"direction": (dx, dy), "strike": bool}`. Omitted players stand still. The engine gives you a **team-normalised** state (your team always attacks toward increasing X); you never handle screen rotation yourself. There are a few examples of bots in the project to refer to. How you decide to generate those direction and strike outputs is completely up to you. You can use a rules-based system or build any model to make those decisions.

2. **Register a team** — import your class in `src/miniball/teams.py` and append a `Team(name="...", ai=YourAI)` entry to `teams_list` (and optionally export the class from `src/miniball/ai/__init__.py`). Named teams are used by `uv run -m miniball.simulation`, the UI team picker, and `simulate_league()`.

3. **Learn the API** — `GameState`, helpers on `PlayerState` and `BallState` (distances, intercept times, path tracing), and action semantics are documented in [`src/miniball/ai/interface.py`](src/miniball/ai/interface.py). `StationaryAI`, `BallChasersAI` and `BaselineAI` are reference implementations that move up in complexity from minimal to non-trivial.

Minimal skeleton:

```python
from miniball.ai import BaseAI, GameState, TeamActions

class MyAI(BaseAI):
    def get_actions(self, state: GameState) -> TeamActions:
        return {
            p.number: {"direction": (0.0, 0.0), "strike": False}
            for p in state.team
        }
```

---


[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
