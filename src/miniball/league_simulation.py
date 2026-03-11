"""Round-robin league simulation.

Runs every ordered pair (home, away) of the supplied teams as a headless
simulation and returns a league-table DataFrame.

Library usage
-------------
::

    from miniball.league_simulation import simulate_league, simulate_fixtures, build_league_table
    from miniball.teams import Team, teams_list
    from miniball.ai import BaselineAI

    # Full league with all registered teams (default)
    table = simulate_league()

    # Subset of registered teams
    from miniball.teams import teams
    table = simulate_league([teams["Baseline (1-2-2)"], teams["Ball Chasers"]])

    # Teams generated on the fly
    custom = Team(name="My AI", ai=BaselineAI)
    table = simulate_league([custom, *teams_list])

    # Lower-level: run specific fixtures and aggregate yourself
    results = simulate_fixtures([(teams_list[0], teams_list[1])])
    table = build_league_table(results)

CLI usage
---------
::

    uv run python -m miniball.league_simulation
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import permutations

import polars as pl

from miniball.teams import Team

# ── Public types ──────────────────────────────────────────────────────────────


@dataclass
class MatchResult:
    """Outcome of a single simulated match."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


# ── Worker function (must be module-level for multiprocessing pickling) ───────


def _run_match(home_team: Team, away_team: Team) -> MatchResult:
    """Run one headless simulation and return a compact result.

    ``Team`` objects (and their ``BaseAI`` instances) are picklable, so they
    cross the process boundary without issue.  Heavy imports happen inside the
    worker to keep the fork payload small.
    """
    from miniball import match_stats
    from miniball.match_simulation import MatchSimulation

    sim = MatchSimulation(home_team, away_team)
    df = sim.simulate_match()
    assert df is not None, "match DataFrame should be populated after game over"
    summary = match_stats.team_summary(df)
    home_row = summary.filter(pl.col("is_home")).row(0, named=True)
    away_row = summary.filter(~pl.col("is_home")).row(0, named=True)

    return MatchResult(
        home_team=home_team.name,
        away_team=away_team.name,
        home_goals=int(home_row["goals"]),
        away_goals=int(away_row["goals"]),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def simulate_fixtures(
    fixtures: list[tuple[Team, Team]],
    *,
    n_workers: int | None = None,
    show_progress: bool = False,
) -> list[MatchResult]:
    """Run a list of ``(home_team, away_team)`` fixtures in parallel.

    Parameters
    ----------
    fixtures:
        Ordered pairs of ``Team`` objects.  Each pair is one match; include
        both orderings to give each team a home fixture.
    n_workers:
        Number of worker processes.  Defaults to ``min(cpu_count, len(fixtures))``.
    show_progress:
        When ``True``, print a one-line result as each match completes.

    Returns
    -------
    list[MatchResult]
        Results in completion order (not fixture order).
    """
    workers = min(n_workers or (os.cpu_count() or 1), len(fixtures))
    results: list[MatchResult] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_fixture = {
            executor.submit(_run_match, home, away): (home, away)
            for home, away in fixtures
        }
        for i, future in enumerate(as_completed(future_to_fixture), 1):
            home, away = future_to_fixture[future]
            r = future.result()
            results.append(r)
            if show_progress:
                score = f"{r.home_goals}–{r.away_goals}"
                print(
                    f"  [{i:2d}/{len(fixtures)}]  {home.name:<32s}  {score:^5s}  {away.name}"
                )

    return results


def build_league_table(results: list[MatchResult]) -> pl.DataFrame:
    """Aggregate a list of match results into a standard league table.

    Columns: pos, team, played, won, drawn, lost, gf, ga, gd, pts

    Rows are sorted by pts → gd → gf (descending).
    """
    rows: list[dict[str, object]] = []
    for r in results:
        home_win = r.home_goals > r.away_goals
        draw = r.home_goals == r.away_goals
        away_win = r.away_goals > r.home_goals

        rows.append(
            {
                "team": r.home_team,
                "played": 1,
                "won": int(home_win),
                "drawn": int(draw),
                "lost": int(away_win),
                "gf": r.home_goals,
                "ga": r.away_goals,
            }
        )
        rows.append(
            {
                "team": r.away_team,
                "played": 1,
                "won": int(away_win),
                "drawn": int(draw),
                "lost": int(home_win),
                "gf": r.away_goals,
                "ga": r.home_goals,
            }
        )

    return (
        pl.DataFrame(rows)
        .group_by("team")
        .agg(
            pl.col("played").sum(),
            pl.col("won").sum(),
            pl.col("drawn").sum(),
            pl.col("lost").sum(),
            pl.col("gf").sum(),
            pl.col("ga").sum(),
        )
        .with_columns(
            (pl.col("gf") - pl.col("ga")).alias("gd"),
            (pl.col("won") * 3 + pl.col("drawn")).alias("pts"),
        )
        .sort(["pts", "gd", "gf"], descending=True)
        .with_row_index(name="pos", offset=1)
        .select(
            [
                "pos",
                "team",
                "played",
                "won",
                "drawn",
                "lost",
                "gf",
                "ga",
                "gd",
                "pts",
            ]
        )
    )


def simulate_league(
    teams: list[Team] | None = None,
    *,
    n_workers: int | None = None,
    show_progress: bool = False,
) -> pl.DataFrame:
    """Run a full round-robin league and return the league table.

    Every ordered pair of teams plays one match (so each pair meets home
    *and* away).  Matches run in parallel using ``ProcessPoolExecutor``.

    Parameters
    ----------
    teams:
        ``Team`` objects to include.  Any ``Team`` is accepted — teams do not
        need to be registered in ``miniball.teams.teams_list``.  Defaults to
        all teams in ``miniball.teams.teams_list``.
    n_workers:
        Number of worker processes.  Defaults to ``min(cpu_count, fixtures)``.
    show_progress:
        When ``True``, print a one-line result as each match completes.

    Returns
    -------
    pl.DataFrame
        League table with columns:
        pos, team, played, won, drawn, lost, gf, ga, gd, pts
    """
    from miniball.teams import teams_list

    teams = teams if teams is not None else teams_list

    fixtures = list(permutations(teams, 2))
    results = simulate_fixtures(
        fixtures, n_workers=n_workers, show_progress=show_progress
    )
    return build_league_table(results)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_league_table(table: pl.DataFrame) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    rt = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        row_styles=None,
    )
    rt.add_column("#", justify="right", style="bold cyan", width=3)
    rt.add_column("Team", justify="left", style="white", min_width=20)
    rt.add_column("P", justify="right")
    rt.add_column("W", justify="right", style="green")
    rt.add_column("D", justify="right", style="yellow")
    rt.add_column("L", justify="right", style="red")
    rt.add_column("GF", justify="right")
    rt.add_column("GA", justify="right")
    rt.add_column("GD", justify="right")
    rt.add_column("Pts", justify="right", style="bold magenta")

    for row in table.iter_rows(named=True):
        gd = row["gd"]
        gd_str = f"+{gd}" if isinstance(gd, int) and gd > 0 else str(gd)
        rt.add_row(
            str(row["pos"]),
            str(row["team"]),
            str(row["played"]),
            str(row["won"]),
            str(row["drawn"]),
            str(row["lost"]),
            str(row["gf"]),
            str(row["ga"]),
            gd_str,
            str(row["pts"]),
        )

    Console().print(rt)


if __name__ == "__main__":
    from rich.console import Console

    from miniball.teams import teams_list

    console = Console()

    _names = [t.name for t in teams_list]
    _fixtures = list(permutations(_names, 2))
    _workers = min(os.cpu_count() or 1, len(_fixtures))

    console.print(
        f"[bold]League:[/bold] {len(_names)} teams · "
        f"{len(_fixtures)} fixtures · {_workers} workers\n"
    )

    _t0 = time.perf_counter()
    _table = simulate_league(show_progress=True)
    _elapsed = time.perf_counter() - _t0

    console.print(
        f"\n[dim]{len(_fixtures)} matches in {_elapsed:.1f} s"
        f"  ({_elapsed / len(_fixtures) * 1000:.0f} ms/match)[/dim]"
    )
    console.rule("[bold]League Table[/bold]")
    _print_league_table(_table)
