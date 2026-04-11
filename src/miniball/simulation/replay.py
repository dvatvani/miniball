"""Frame reconstruction from parquet match data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from miniball.ai import (
    BallState,
    GameState,
    MatchState,
    PlayerState,
    TeamActions,
)
from miniball.config import STANDARD_PITCH_HEIGHT as _H
from miniball.config import STANDARD_PITCH_WIDTH as _W

if TYPE_CHECKING:
    from miniball.simulation.runner import MatchResult


@dataclass
class FrameSnapshot:
    """A fully reconstructed single frame from a match parquet file.

    Both team perspectives are provided so that AI logic, rich player/ball
    methods, and coordinate helpers all work directly without re-parsing
    tabular data.

    Attributes
    ----------
    frame_number:
        Zero-based index matching the ``frame_number`` column in the parquet.
    match_time_seconds:
        Elapsed game time in seconds since kick-off.
    state_a, state_b:
        ``GameState`` from team A's (home) and team B's (away) perspective
        respectively.  Both use the standard team frame (attacking right).
    actions_a, actions_b:
        The effective ``TeamActions`` submitted by each team's AI for this
        frame, in each team's own normalised coordinate frame.
    """

    frame_number: int
    match_time_seconds: float
    state_a: GameState
    state_b: GameState
    actions_a: TeamActions
    actions_b: TeamActions


def reconstruct_frames(df: pl.DataFrame) -> list[FrameSnapshot]:
    """Reconstruct per-frame native objects from a match DataFrame.

    The returned ``FrameSnapshot`` list mirrors what the simulation engine
    produces internally each frame, making it straightforward to replay a
    saved match, inspect AI decisions, or run analytical tools that operate
    on ``GameState`` / ``PlayerState`` / ``BallState`` objects.

    Parameters
    ----------
    df:
        polars DataFrame produced by ``MatchSimulation.build_match_df``.

    Returns
    -------
    list[FrameSnapshot]
        One entry per frame, sorted by ``frame_number``.
    """
    snapshots: list[FrameSnapshot] = []

    for frame_df in df.sort("frame_number").partition_by(
        "frame_number", maintain_order=True
    ):
        home_rows = frame_df.filter(pl.col("is_home"))
        away_rows = frame_df.filter(~pl.col("is_home"))

        first_home = home_rows.row(0, named=True)
        first_away = away_rows.row(0, named=True)
        frame_number: int = first_home["frame_number"]
        match_time: float = first_home["match_time_seconds"]

        # ── Build team A's GameState (home perspective = global frame) ────────
        team_a_players = [
            PlayerState(
                number=row["player_number"],
                is_teammate=True,
                is_home=True,
                has_ball=row["has_ball"],
                cooldown_timer=row["cooldown_timer"],
                location=(row["player_x"], row["player_y"]),
            )
            for row in home_rows.iter_rows(named=True)
        ]
        # Away players in team A's frame: rotate team-B coords back to global.
        opp_for_a = [
            PlayerState(
                number=row["player_number"],
                is_teammate=False,
                is_home=False,
                has_ball=row["has_ball"],
                cooldown_timer=row["cooldown_timer"],
                location=(_W - row["player_x"], _H - row["player_y"]),
            )
            for row in away_rows.iter_rows(named=True)
        ]
        state_a = GameState(
            team=team_a_players,
            opposition=opp_for_a,
            ball=BallState(
                location=(first_home["ball_x"], first_home["ball_y"]),
                velocity=(first_home["ball_vx"], first_home["ball_vy"]),
            ),
            match_state=MatchState(
                team_current_score=first_home["team_score"],
                opposition_current_score=first_home["opposition_score"],
                match_time_seconds=match_time,
            ),
            is_home=True,
        )

        # ── Build team B's GameState (away perspective) ───────────────────────
        team_b_players = [
            PlayerState(
                number=row["player_number"],
                is_teammate=True,
                is_home=False,
                has_ball=row["has_ball"],
                cooldown_timer=row["cooldown_timer"],
                location=(row["player_x"], row["player_y"]),
            )
            for row in away_rows.iter_rows(named=True)
        ]
        # Home players in team B's frame: rotate global coords to team-B frame.
        opp_for_b = [
            PlayerState(
                number=row["player_number"],
                is_teammate=False,
                is_home=True,
                has_ball=row["has_ball"],
                cooldown_timer=row["cooldown_timer"],
                location=(_W - row["player_x"], _H - row["player_y"]),
            )
            for row in home_rows.iter_rows(named=True)
        ]
        state_b = GameState(
            team=team_b_players,
            opposition=opp_for_b,
            ball=BallState(
                location=(first_away["ball_x"], first_away["ball_y"]),
                velocity=(first_away["ball_vx"], first_away["ball_vy"]),
            ),
            match_state=MatchState(
                team_current_score=first_away["team_score"],
                opposition_current_score=first_away["opposition_score"],
                match_time_seconds=match_time,
            ),
            is_home=False,
        )

        actions_a: TeamActions = {
            row["player_number"]: {
                "direction": (row["action_dx"], row["action_dy"]),
                "strike": row["strike"],
            }
            for row in home_rows.iter_rows(named=True)
        }
        actions_b: TeamActions = {
            row["player_number"]: {
                "direction": (row["action_dx"], row["action_dy"]),
                "strike": row["strike"],
            }
            for row in away_rows.iter_rows(named=True)
        }

        snapshots.append(
            FrameSnapshot(
                frame_number=frame_number,
                match_time_seconds=match_time,
                state_a=state_a,
                state_b=state_b,
                actions_a=actions_a,
                actions_b=actions_b,
            )
        )

    return snapshots


def load_match(
    path: str | Path,
) -> tuple[MatchResult, pl.DataFrame, list[FrameSnapshot]]:
    """Load a match from a parquet file and return result, DataFrame, and snapshots."""
    from miniball.simulation.runner import _df_to_match_result

    df = pl.read_parquet(path)
    snapshots = reconstruct_frames(df)
    match_result = _df_to_match_result(df)

    return match_result, df, snapshots
