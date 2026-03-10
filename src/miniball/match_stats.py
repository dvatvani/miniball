"""Post-match statistics derived from the frame-level history DataFrame.

All functions accept the DataFrame produced by ``FootballGame._build_match_df``
and return tidy Polars DataFrames suitable for display or further analysis.

Possession sessionisation
─────────────────────────
``sessionise_possessions`` identifies possession *sessions*: contiguous runs of
frames during which the same team controls the ball.  Brief free-ball phases
that occur between two touches by the same team (e.g. a pass in flight) are
bridged automatically because those frames simply don't appear in the filtered
dataset.

Call ``possession_stats`` on the output to obtain aggregated per-team metrics.
"""

from __future__ import annotations

import polars as pl


def avg_positions(df: pl.DataFrame) -> pl.DataFrame:
    """Return each player's average position over the match in global coordinates.

    Columns
    -------
    team, is_home, player_number, avg_x, avg_y
        Positions are in the global pitch frame (home team attacks right,
        x ∈ [0, 120], y ∈ [0, 80]).
    """
    return (
        df.group_by(["team", "is_home", "player_number"])
        .agg(
            pl.col("pos_x_global").mean().alias("avg_x"),
            pl.col("pos_y_global").mean().alias("avg_y"),
        )
        .sort(["team", "player_number"])
    )


def team_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Return high-level per-team match statistics.

    Columns
    -------
    team, is_home, goals, goals_against, shots,
    possession_pct, possession_count, avg_duration

    Notes
    -----
    ``shots`` counts frames where a player had the ball and requested a shot.
    Because the ball is released the same frame a shot fires, this reliably
    gives one count per shot event.

    ``possession_pct`` is derived from ``possession_stats(sessionise_possessions(df))``
    and represents each team's share of total possessed time (sums to 100 %).
    ``possession_count`` and ``avg_duration`` (seconds) come from the same source.
    """
    base = (
        df.group_by(["team", "is_home"])
        .agg(
            pl.col("team_score").max().alias("goals"),
            pl.col("opposition_score").max().alias("goals_against"),
            (pl.col("shoot") & pl.col("has_ball")).sum().alias("shots"),
        )
        .sort("is_home", descending=True)
    )

    poss = possession_stats(sessionise_possessions(df))

    return base.join(
        poss.select(
            ["team", "is_home", "possession_pct", "possession_count", "avg_duration"]
        ),
        on=["team", "is_home"],
        how="left",
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
        Frame-level DataFrame produced by ``FootballGame._build_match_df``.

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
