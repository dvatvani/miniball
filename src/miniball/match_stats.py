"""Post-match statistics derived from the frame-level history DataFrame.

All functions accept the DataFrame produced by ``MatchSimulation.build_match_df``
and return tidy Polars DataFrames suitable for display or further analysis.

Possession sessionisation
─────────────────────────
``sessionise_possessions`` identifies possession *sessions*: contiguous runs of
frames during which the same team controls the ball.  Brief free-ball phases
that occur between two touches by the same team (e.g. a pass in flight) are
bridged automatically because those frames simply don't appear in the filtered
dataset.

Call ``possession_stats`` on the output to obtain aggregated per-team metrics.

Strike classification
─────────────────────
``strike_stats`` classifies each strike event as a **shot** or a **pass**:

* **Shot** — struck from the attacking half (team-frame x > 60) with a
  trajectory that would cross the goal line (x = 120) within *twice* the goal
  height.  This intentionally includes near-misses so that all genuine attempts
  on goal are captured regardless of accuracy.
* **Pass attempt** — any strike that is not a shot.

Each category is further refined:

* **Shot on target** — trajectory crosses within the actual goal bounds.
* **Successful pass** — the next player to gain possession after the strike
  is a teammate of the striker.

Because ball drag is isotropic (identical drag on ``vx`` and ``vy``), the
direction of travel is perfectly preserved over time, so the y-intercept at
x = 120 reduces to simple linear extrapolation regardless of speed or drag.
"""

from __future__ import annotations

import polars as pl

from miniball.config import (
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)


def avg_positions(df: pl.DataFrame) -> pl.DataFrame:
    """Return each player's average position over the match in team coordinates.

    Coordinates are in the **team's own normalised frame** (the team always
    attacks right, x ∈ [0, 120], y ∈ [0, 80]).  This makes the output
    consistent across multiple matches regardless of which physical side each
    team occupied.

    Columns
    -------
    team, is_home, player_number, avg_x, avg_y
    """
    return (
        df.group_by(["team", "is_home", "player_number"])
        .agg(
            pl.col("pos_x").mean().alias("avg_x"),
            pl.col("pos_y").mean().alias("avg_y"),
        )
        .sort(["team", "player_number"])
    )


def annotate_strikes(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per strike event, enriched with classification metadata.

    Parameters
    ----------
    df:
        Frame-level DataFrame produced by ``MatchSimulation.build_match_df``.

    Returns
    -------
    pl.DataFrame
        One row per strike, sorted by ``frame_number``, with columns:

        frame_number, team, is_home,
        ball_x, ball_y, action_dx, action_dy,
        y_at_goal          – projected y at the goal line (null if not heading forward),
        is_shot            – True when struck from attacking half toward the goal mouth,
        is_shot_on_target  – True when the trajectory crosses within actual goal bounds,
        is_goal            – True when the shot is followed by a score increase before the next strike,
        is_pass            – True for all non-shot strikes,
        is_pass_successful – True when the next player to gain possession is a teammate

    Notes
    -----
    All coordinates are in **team perspective** (team always attacks right,
    goal at x = 120).  The ``ball_x``/``ball_y`` and ``action_dx``/``action_dy``
    columns in the history DataFrame already carry this convention.
    """
    goal_half_h = STANDARD_GOAL_HEIGHT / 2
    goal_center_y = STANDARD_PITCH_HEIGHT / 2
    goal_lo = goal_center_y - goal_half_h
    goal_hi = goal_center_y + goal_half_h
    shot_y_lo = goal_center_y - 2 * goal_half_h
    shot_y_hi = goal_center_y + 2 * goal_half_h
    pitch_mid = STANDARD_PITCH_WIDTH / 2  # 60.0

    _empty_schema: dict[str, type[pl.DataType]] = {
        "frame_number": pl.Int64,
        "team": pl.String,
        "is_home": pl.Boolean,
        "ball_x": pl.Float64,
        "ball_y": pl.Float64,
        "action_dx": pl.Float64,
        "action_dy": pl.Float64,
        "y_at_goal": pl.Float64,
        "is_shot": pl.Boolean,
        "is_shot_on_target": pl.Boolean,
        "is_goal": pl.Boolean,
        "is_pass": pl.Boolean,
        "is_pass_successful": pl.Boolean,
    }

    # ── Extract one row per strike event ──────────────────────────────────────
    events = (
        df.filter(pl.col("strike") & pl.col("has_ball"))
        .select(
            [
                "frame_number",
                "team",
                "is_home",
                "ball_x",
                "ball_y",
                "action_dx",
                "action_dy",
            ]
        )
        .sort("frame_number")
    )

    if events.is_empty():
        return pl.DataFrame(schema=_empty_schema)

    # ── Shot / pass classification ─────────────────────────────────────────────
    # Because drag acts identically on vx and vy the ball travels in a straight
    # line, so the y-intercept at the goal line is purely linear.
    events = (
        events.with_columns(
            pl.when(pl.col("action_dx") > 1e-6)
            .then(
                pl.col("ball_y")
                + (pl.col("action_dy") / pl.col("action_dx"))
                * (STANDARD_PITCH_WIDTH - pl.col("ball_x"))
            )
            .otherwise(None)
            .alias("y_at_goal"),
        )
        .with_columns(
            (
                (pl.col("ball_x") > pitch_mid)
                & (pl.col("action_dx") > 1e-6)
                & pl.col("y_at_goal").is_not_null()
                & (pl.col("y_at_goal") >= shot_y_lo)
                & (pl.col("y_at_goal") <= shot_y_hi)
            ).alias("is_shot"),
        )
        .with_columns(
            (
                pl.col("is_shot")
                & (pl.col("y_at_goal") >= goal_lo)
                & (pl.col("y_at_goal") <= goal_hi)
            ).alias("is_shot_on_target"),
            (~pl.col("is_shot")).alias("is_pass"),
        )
    )

    # ── Pass success: next player to gain possession is a teammate ─────────────
    # Build a lookup table: for each frame number, which team gains possession?
    next_possession = (
        df.filter(pl.col("has_ball"))
        .select(["frame_number", "team"])
        .unique(subset=["frame_number"])
        .sort("frame_number")
        .rename({"frame_number": "next_frame", "team": "next_team"})
    )

    # For each strike at frame F, find the first possession at frame > F.
    # join_asof(strategy="forward") finds the nearest next_frame >= lookup key.
    # Using frame_number + 1 as the key skips possession by the striker themselves
    # on the same frame.
    events = (
        events.with_columns((pl.col("frame_number") + 1).alias("next_frame"))
        .join_asof(next_possession, on="next_frame", strategy="forward")
        .with_columns(
            (
                pl.col("is_pass")
                & pl.col("next_team").is_not_null()
                & (pl.col("next_team") == pl.col("team"))
            ).alias("is_pass_successful"),
        )
        .drop(["next_frame", "next_team"])
    )

    # ── Goal detection ─────────────────────────────────────────────────────────
    # Find frames where each team's score increased (a goal was conceded).
    # Sorted by (team, goal_frame) so join_asof can verify within-group order.
    goal_events = (
        df.select(["frame_number", "team", "team_score"])
        .unique(subset=["frame_number", "team"])
        .sort(["team", "frame_number"])
        .with_columns(pl.col("team_score").shift(1).over("team").alias("prev_score"))
        .filter(pl.col("team_score") > pl.col("prev_score").fill_null(0))
        .select(["frame_number", "team"])
        .rename({"frame_number": "goal_frame"})
        .sort(["team", "goal_frame"])
    )

    # For each shot at frame F by team T, find the next goal by T after F
    # (join_asof by="team" searches within the same team's goal events).
    # next_event_frame caps the window: the goal must precede the next strike.
    #
    # next_event_frame is computed first (requires global frame order via
    # shift(-1)), then events are re-sorted by (team, goal_lookup) so that
    # Polars can verify within-group sort order for the by="team" join_asof.
    events = events.with_columns(
        pl.col("frame_number").shift(-1).alias("next_event_frame"),
        (pl.col("frame_number") + 1).alias("goal_lookup"),
    )
    return (
        events.sort(["team", "goal_lookup"])
        .join_asof(
            goal_events,
            left_on="goal_lookup",
            right_on="goal_frame",
            by="team",
            strategy="forward",
        )
        .with_columns(
            (
                pl.col("is_shot")
                & pl.col("goal_frame").is_not_null()
                & (
                    pl.col("next_event_frame").is_null()
                    | (pl.col("goal_frame") <= pl.col("next_event_frame"))
                )
            ).alias("is_goal"),
        )
        .drop(["goal_lookup", "goal_frame", "next_event_frame"])
        .sort("frame_number")
    )


def strike_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Classify every strike as a shot or a pass and aggregate per-team stats.

    Parameters
    ----------
    df:
        Frame-level DataFrame produced by ``MatchSimulation.build_match_df``.

    Returns
    -------
    pl.DataFrame
        One row per team with columns:

        team, is_home,
        passes, passes_completed, pass_accuracy (0–100 %),
        shots, shots_on_target, shot_accuracy (0–100 %)
    """
    _empty_schema: dict[str, type[pl.DataType]] = {
        "team": pl.String,
        "is_home": pl.Boolean,
        "passes": pl.Int32,
        "passes_completed": pl.Int32,
        "pass_accuracy": pl.Float64,
        "shots": pl.Int32,
        "shots_on_target": pl.Int32,
        "shot_accuracy": pl.Float64,
    }

    events = annotate_strikes(df)

    if events.is_empty():
        return pl.DataFrame(schema=_empty_schema)

    return (
        events.group_by(["team", "is_home"])
        .agg(
            pl.col("is_pass").sum().alias("passes"),
            pl.col("is_pass_successful").sum().alias("passes_completed"),
            pl.col("is_shot").sum().alias("shots"),
            pl.col("is_shot_on_target").sum().alias("shots_on_target"),
        )
        .with_columns(
            pl.when(pl.col("passes") > 0)
            .then((pl.col("passes_completed") / pl.col("passes") * 100).round(1))
            .otherwise(0.0)
            .alias("pass_accuracy"),
            pl.when(pl.col("shots") > 0)
            .then((pl.col("shots_on_target") / pl.col("shots") * 100).round(1))
            .otherwise(0.0)
            .alias("shot_accuracy"),
        )
        .sort("is_home", descending=True)
    )


def team_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Return high-level per-team match statistics.

    Columns
    -------
    team, is_home, goals, goals_against,
    passes, passes_completed, pass_accuracy,
    shots, shots_on_target, shot_accuracy, shot_conversion_rate,
    possession_pct, possession_count, avg_duration

    Notes
    -----
    ``possession_pct`` is derived from ``possession_stats(sessionise_possessions(df))``
    and represents each team's share of total possessed time (sums to 100 %).
    ``possession_count`` and ``avg_duration`` (seconds) come from the same source.

    ``shot_conversion_rate`` is ``goals / shots * 100`` (0–100 %).
    ``pass_accuracy`` and ``shot_accuracy`` are also expressed as 0–100 %.
    """
    base = (
        df.group_by(["team", "is_home"])
        .agg(
            pl.col("team_score").max().alias("goals"),
            pl.col("opposition_score").max().alias("goals_against"),
        )
        .sort("is_home", descending=True)
    )

    poss = possession_stats(sessionise_possessions(df))
    strikes = strike_stats(df)

    return (
        base.join(
            poss.select(
                [
                    "team",
                    "is_home",
                    "possession_pct",
                    "possession_count",
                    "avg_duration",
                ]
            ),
            on=["team", "is_home"],
            how="left",
        )
        .join(
            strikes.select(
                [
                    "team",
                    "is_home",
                    "passes",
                    "passes_completed",
                    "pass_accuracy",
                    "shots",
                    "shots_on_target",
                    "shot_accuracy",
                ]
            ),
            on=["team", "is_home"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("shots") > 0)
            .then((pl.col("goals") / pl.col("shots") * 100).round(1))
            .otherwise(0.0)
            .alias("shot_conversion_rate"),
        )
    )


def sessionise_possessions(df: pl.DataFrame) -> pl.DataFrame:
    """Identify possession sessions from the frame-level match data.

    A possession starts when a team first gains control of the ball and ends
    on the last frame that same team controls it before the opposition gains
    control (or the match ends).  Brief free-ball phases between two touches
    by the *same* team are bridged automatically: those frames are absent from
    the filtered dataset so they don't break the run.

    Parameters
    ----------
    df:
        Frame-level DataFrame produced by ``MatchSimulation.build_match_df``.

    Returns
    -------
    pl.DataFrame
        One row per possession with columns:

        possession_index  int     – chronological counter, starting at 1
        team              str     – name of the team in possession
        is_home           bool    – True for the home team
        start_frame       int     – first frame of the possession
        end_frame         int     – last  frame of the possession
        start_time        float   – match_time_seconds at start_frame
        end_time          float   – match_time_seconds at end_frame
        duration          float   – end_time − start_time (seconds)
    """
    # One row per frame where any player has the ball.
    possessed = (
        df.filter(pl.col("has_ball"))
        .select(["frame_number", "match_time_seconds", "team", "is_home"])
        .sort("frame_number")
    )

    if possessed.is_empty():
        return pl.DataFrame(
            schema={
                "possession_index": pl.Int32,
                "team": pl.String,
                "is_home": pl.Boolean,
                "start_frame": pl.Int64,
                "end_frame": pl.Int64,
                "start_time": pl.Float64,
                "end_time": pl.Float64,
                "duration": pl.Float64,
            }
        )

    # Increment the possession counter each time the possessing team changes.
    # fill_null(True) ensures the very first row always opens possession #1.
    possessed = possessed.with_columns(
        (pl.col("team") != pl.col("team").shift(1))
        .fill_null(True)
        .cast(pl.Int32)
        .cum_sum()
        .alias("possession_index")
    )

    return (
        possessed.group_by("possession_index")
        .agg(
            pl.col("team").first(),
            pl.col("is_home").first(),
            pl.col("frame_number").min().alias("start_frame"),
            pl.col("frame_number").max().alias("end_frame"),
            pl.col("match_time_seconds").min().alias("start_time"),
            pl.col("match_time_seconds").max().alias("end_time"),
        )
        .with_columns((pl.col("end_time") - pl.col("start_time")).alias("duration"))
        .sort("possession_index")
    )


def possession_stats(possessions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate possession-level data to per-team summary statistics.

    Parameters
    ----------
    possessions:
        Output of ``sessionise_possessions``.

    Returns
    -------
    pl.DataFrame
        One row per team with columns:

        team                    str    – team name
        is_home                 bool
        possession_count        int    – number of separate possessions
        total_possession_time   float  – total seconds in possession
        possession_pct          float  – share of total possessed time (sums to 100 %)
        avg_duration            float  – mean possession duration (s)
        median_duration         float  – median possession duration (s)
        max_duration            float  – longest single possession (s)

    Notes
    -----
    ``possession_pct`` is normalised to the total time *any* team held the ball,
    so it always sums to 100 % across both teams.  Free-ball time is excluded.
    """
    stats = (
        possessions.group_by(["team", "is_home"])
        .agg(
            pl.len().alias("possession_count"),
            pl.col("duration").sum().alias("total_possession_time"),
            pl.col("duration").mean().alias("avg_duration"),
            pl.col("duration").median().alias("median_duration"),
            pl.col("duration").max().alias("max_duration"),
        )
        .sort("is_home", descending=True)
    )

    raw_total = stats["total_possession_time"].sum()
    assert isinstance(raw_total, float)

    if raw_total > 0:
        stats = stats.with_columns(
            (pl.col("total_possession_time") / raw_total * 100)
            .round(1)
            .alias("possession_pct")
        )
    else:
        stats = stats.with_columns(pl.lit(50.0).alias("possession_pct"))

    return stats
