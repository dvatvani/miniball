"""Batch / parallel match execution and result aggregation."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from rich.console import Console

from miniball.teams import Team

console = Console()


@dataclass
class MatchResult:
    """Outcome of a single simulated match."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    filename: str | None = None


def _df_to_match_result(df: pl.DataFrame) -> MatchResult:
    """Convert a match DataFrame to a MatchResult."""
    home_team_name = df.filter(pl.col("is_home")).row(0, named=True)["team_name"]
    away_team_name = df.filter(~pl.col("is_home")).row(0, named=True)["team_name"]
    last = df.filter(pl.col("is_home")).tail(1).row(0, named=True)
    return MatchResult(
        home_team_name,
        away_team_name,
        int(last["team_score"]),
        int(last["opposition_score"]),
    )


def _simulate_match(
    home_team: Team, away_team: Team, save_data: bool = False
) -> MatchResult:
    """Run one headless simulation and return a compact result.

    ``Team`` objects (and their ``BaseAI`` instances) are picklable, so they
    cross the process boundary without issue.
    """
    from miniball.simulation.engine import MatchSimulation
    from miniball.simulation.recording import write_parquet

    sim = MatchSimulation(home_team, away_team)
    df = sim.simulate_match()
    assert df is not None, "match DataFrame should be populated after game over"

    match_result = _df_to_match_result(df)
    if save_data:
        path = write_parquet(df, home_team.name, away_team.name, verbose=False)
        match_result.filename = str(path)
    return match_result


def simulate_matches(
    matches: list[tuple[Team, Team]],
    *,
    n_workers: int | None = None,
    show_progress: bool = False,
    save_data: bool = False,
) -> list[MatchResult]:
    """Run a list of ``(home_team, away_team)`` fixtures in parallel.

    Parameters
    ----------
    matches:
        Ordered pairs of ``Team`` objects.  Each pair is one match; include
        both orderings to give each team a home fixture.
    n_workers:
        Number of worker processes.  Defaults to ``min(cpu_count, len(fixtures))``.
    show_progress:
        When ``True``, print a one-line result as each match completes.
    save_data:
        When ``True``, save the match data to parquet files.
    Returns
    -------
    list[MatchResult]
        Results in completion order (not fixture order).
    """
    workers = min(n_workers or (os.cpu_count() or 1), len(matches))
    results: list[MatchResult] = []
    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_fixture = {
            executor.submit(_simulate_match, home, away, save_data): (
                home,
                away,
                save_data,
            )
            for home, away in matches
        }
        for i, future in enumerate(as_completed(future_to_fixture), 1):
            home, away, save_data = future_to_fixture[future]
            r = future.result()
            results.append(r)
            if show_progress:
                score = f"{r.home_goals}–{r.away_goals}"
                print(
                    f"  [{i:2d}/{len(matches)}]  {home.name:<32s}  {score:^5s}  {away.name}"
                )

    console.print(
        f"\n[dim]{len(matches)} matches in {time.perf_counter() - start_time:.1f} s"
        f"  ({(time.perf_counter() - start_time) / len(matches) * 1000:.0f} ms/match)[/dim]"
    )
    if save_data:
        console.print(
            f"\n[dim]{len(matches)} matches saved to parquet files in {Path('match_data').absolute()}[/dim]"
        )
    return results
