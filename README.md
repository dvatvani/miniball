# Miniball

A developer/analyst-friendly 2D 5-as-side football-inspired game.

The game can be played as an arcade game with a controller, but a big part of the game's design and appeal is to enable you to program your own bots and have them play against each other. The rules of the game have also deliberately been kept simple to make it easier to model. After playing a few games (or watching the bots play each other), you're encouraged to build your own AI team to see where it would place in a all-AI league.

### Gameplay / Rules
 - 5-a-side
 - 2D for modeling simplicity
 - Controls are completely manual i.e. no shot / pass assist.
 - No player inertia / acceleration. Players can stay still or move at a fixed speed in that direction.
 - When the ball collides with a player, it snaps to the player and moves with that player.
 - The defending team can collide with the player on the ball to automatically tackle and regain possession
 - Other than moving, there's a single action that can be taken to strike the ball in the direction the player is moving. This happens instantaneously (no charge-up) and at a fixed speed. This action, combined with the movement direction, is used to pass, clear and shoot the ball.
 - To prevent some exploits that emerge from the rules above, players that lose or strike the ball have a short cooldown period before they can interact with the ball again
 - No offsides, fouls or set pieces. There are walls along the edge of the pitch to keep the ball in play.
- Kickoffs: The furthest forward player in the home team starts with possession rather than kick-odds happening at the center circle. After a goal is conceded, the conceding team kicks off in a similar way.
- Player collisions: Players positions can't overlap. Players will instead push other players (not at full running speed) if attempting to move a location containing another player.

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
uv run -m miniball.match_simulation
```

The results of the match simulation are stored in the `match_data` directory, and can be analysed with the `notebooks/analyse_match.py` marimo notebook.

Matches can also be simulated easily from within Python:

```python
from miniball.teams import teams_list
from miniball.match_simulation import MatchSimulation
sim = MatchSimulation(teams_list[0], teams_list[1])
match_df = sim.simulate_match()
sim.export_history(). # Optionally, export the data to parquet
```

### League simulations

Round-robin league simulations including all AI models can be run from the CLI
```bash
uv run -m miniball.league_simulation
```

League simulations can also be easily run from within Python:

```python
from miniball.league_simulation import simulate_league
league_table_df = simulate_league()  # teams to include can be passed in. Uses all by default
```


---

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
