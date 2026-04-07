import random

import typer

from miniball.simulation.runner import simulate_matches
from miniball.teams import teams, teams_list


def _cli(
    home_team: str | None = typer.Option(None, help="Home team model name"),
    away_team: str | None = typer.Option(None, help="Away team model name"),
    save_data: bool = typer.Option(False, help="Save match data to parquet files"),
    n_matches: int = typer.Option(1, help="Number of matches to simulate"),
) -> None:
    matches = []
    for _ in range(n_matches):
        h = teams[home_team] if home_team else random.choice(teams_list)
        a = (
            teams[away_team]
            if away_team
            else random.choice([t for t in teams_list if t is not h])
        )
        matches.append((h, a))
    simulate_matches(matches, show_progress=True, save_data=save_data)


typer.run(_cli)
