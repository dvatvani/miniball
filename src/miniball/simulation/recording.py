"""Match recording and parquet serialisation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from miniball.ai import GameState, PlayerAction, TeamActions
from miniball.coords import global_delta_to_team, global_to_team


@dataclass
class FrameRecord:
    """Single-frame snapshot used for post-game analysis.

    ``state`` is expressed in team A's coordinate system (team A always
    attacks right), which serves as the global pitch reference frame.

    ``actions_team_a`` and ``actions_team_b`` are each in their own team's
    normalised frame (both teams attack right in their respective frames).
    To convert team B's directions to the global frame, negate both components.

    Human input overrides are already merged in: these are the *effective*
    actions that were sent to the game engine, not the raw AI outputs.
    """

    state: GameState  # global reference frame (team A perspective)
    actions_team_a: TeamActions  # normalised; team A attacks right
    actions_team_b: TeamActions  # normalised; team B's own frame (also attacks right)
    human_player: tuple[bool, int] | None  # (is_home, player_number) or None


def build_rows(
    history: list[FrameRecord],
    name_a: str,
    name_b: str,
) -> list[dict[str, object]]:
    """Flatten frame records into row dicts for a Polars DataFrame."""
    rows: list[dict[str, object]] = []
    for frame_number, record in enumerate(history):
        gbx, gby = record.state.ball.location
        gbvx, gbvy = record.state.ball.velocity
        score_a = record.state.match_state.team_current_score
        score_b = record.state.match_state.opposition_current_score
        match_time = record.state.match_state.match_time_seconds

        _null_action: PlayerAction = {"direction": (0.0, 0.0), "strike": False}
        hp = record.human_player

        for player in record.state.team:  # team A – own frame = global
            num = player.number
            gx, gy = player.location
            pa_a = record.actions_team_a.get(num, _null_action)
            dx, dy = pa_a["direction"]
            rows.append(
                {
                    "frame_number": frame_number,
                    "match_time_seconds": match_time,
                    "team_name": name_a,
                    "opposition_name": name_b,
                    "is_home": True,
                    "player_number": num,
                    "is_human_controlled": hp is not None
                    and hp[0] is True
                    and hp[1] == num,
                    "player_x": gx,
                    "player_y": gy,
                    "has_ball": player.has_ball,
                    "cooldown_timer": player.cooldown_timer,
                    "action_dx": dx,
                    "action_dy": dy,
                    "strike": pa_a["strike"],
                    "ball_x": gbx,
                    "ball_y": gby,
                    "ball_vx": gbvx,
                    "ball_vy": gbvy,
                    "team_score": score_a,
                    "opposition_score": score_b,
                }
            )

        for player in record.state.opposition:  # team B
            num = player.number
            gx, gy = player.location
            bx, by = global_to_team(gx, gy, is_home=False)
            pa_b = record.actions_team_b.get(num, _null_action)
            dx_b, dy_b = pa_b["direction"]
            bbx, bby = global_to_team(gbx, gby, is_home=False)
            bbvx, bbvy = global_delta_to_team(gbvx, gbvy, is_home=False)
            rows.append(
                {
                    "frame_number": frame_number,
                    "match_time_seconds": match_time,
                    "team_name": name_b,
                    "opposition_name": name_a,
                    "is_home": False,
                    "player_number": num,
                    "is_human_controlled": hp is not None
                    and hp[0] is False
                    and hp[1] == num,
                    "player_x": bx,
                    "player_y": by,
                    "has_ball": player.has_ball,
                    "cooldown_timer": player.cooldown_timer,
                    "action_dx": dx_b,
                    "action_dy": dy_b,
                    "strike": pa_b["strike"],
                    "ball_x": bbx,
                    "ball_y": bby,
                    "ball_vx": bbvx,
                    "ball_vy": bbvy,
                    "team_score": score_b,
                    "opposition_score": score_a,
                }
            )

    return rows


def write_parquet(
    df: pl.DataFrame,
    home_team_name: str,
    away_team_name: str,
    verbose: bool = True,
) -> None:
    """Write a match history DataFrame to a timestamped parquet file.

    Narrow dtypes are applied here (not in ``build_match_df``) so that
    in-memory stats operations work with full-precision data while only
    the persisted file uses compact types.
    """
    from miniball.simulation.runner import _df_to_match_result

    out_dir = Path("match_data")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    match_result = _df_to_match_result(df)
    path = (
        out_dir
        / f"match_{timestamp}_{match_result.home_team}_{match_result.away_team}_{match_result.home_goals}-{match_result.away_goals}_{unique_id}.parquet"
    )
    df.with_columns(
        # Spatial / physics – float32
        pl.col("player_x").cast(pl.Float32),
        pl.col("player_y").cast(pl.Float32),
        pl.col("action_dx").cast(pl.Float32),
        pl.col("action_dy").cast(pl.Float32),
        pl.col("ball_x").cast(pl.Float32),
        pl.col("ball_y").cast(pl.Float32),
        pl.col("ball_vx").cast(pl.Float32),
        pl.col("ball_vy").cast(pl.Float32),
        pl.col("cooldown_timer").cast(pl.Float32),
        # Time – float32
        pl.col("match_time_seconds").cast(pl.Float32),
        # Integers – downcast to smallest fitting type
        pl.col("frame_number").cast(pl.Int16),
        pl.col("player_number").cast(pl.Int8),
        pl.col("team_score").cast(pl.Int8),
        pl.col("opposition_score").cast(pl.Int8),
    ).write_parquet(path)
    if verbose:
        n_frames = df["frame_number"].n_unique()
        print(f"Match data saved → {path}  ({n_frames} frames · {len(df)} rows)")
