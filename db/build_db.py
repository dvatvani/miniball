"""Build the miniball DuckDB analytics database with views over match parquet files.

Run from the project root:
    uv run python db/build_db.py

Creates db/miniball.duckdb with views: tracking, events, player_possession,
team_possession, player_match, team_match.
"""

from __future__ import annotations

import glob

import duckdb

from miniball.config import (
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)

DB_PATH = "db/miniball.duckdb"
MATCH_DATA_GLOB = "match_data/*.parquet"

# Pitch geometry from miniball/config.py, embedded as SQL constants.
_PITCH_W = STANDARD_PITCH_WIDTH
_PITCH_H = STANDARD_PITCH_HEIGHT
_GOAL_H = STANDARD_GOAL_HEIGHT
_GOAL_CENTER_Y = _PITCH_H / 2
_GOAL_HALF_H = _GOAL_H / 2
_GOAL_LO = _GOAL_CENTER_Y - _GOAL_HALF_H
_GOAL_HI = _GOAL_CENTER_Y + _GOAL_HALF_H
_SHOT_Y_LO = _GOAL_CENTER_Y - 2 * _GOAL_HALF_H
_SHOT_Y_HI = _GOAL_CENTER_Y + 2 * _GOAL_HALF_H
_PITCH_MID_X = _PITCH_W / 2


def build_db() -> None:
    """Create or rebuild the analytics database."""
    match_files = glob.glob("match_data/*.parquet")
    if not match_files:
        print("No parquet files found in match_data/. Run some matches first.")
        return

    con = duckdb.connect(str(DB_PATH))
    try:
        print(f"Building database from {len(match_files)} match file(s)…")
        _create_tracking(con)
        _create_events(con)
        _create_player_possession(con)
        _create_team_possession(con)
        _create_player_match(con)
        _create_team_match(con)
        _create_human_player_match(con)
        _create_human_team_match(con)

    finally:
        con.close()
    print(f"\nDatabase ready: {DB_PATH}")


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------

_SAFE_GLOB = MATCH_DATA_GLOB.replace("'", "''")


def _create_tracking(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE OR REPLACE VIEW tracking AS
    SELECT
        filename,
        frame_number,
        match_time_seconds,
        team_name,
        opposition_name,
        is_home,
        player_number,
        is_human_controlled,
        player_x,
        player_y,
        has_ball,
        cooldown_timer,
        action_dx,
        action_dy,
        strike,
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        team_score,
        opposition_score
    FROM read_parquet('{_SAFE_GLOB}', filename = true)
    """)


def _create_events(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE OR REPLACE VIEW events AS
    WITH
    -- One row per frame where a player has the ball
    ball_holders AS (
        SELECT
            filename, frame_number, match_time_seconds,
            team_name, player_number, is_home, opposition_name,
            is_human_controlled,
            player_x, player_y,
            ball_x, ball_y, action_dx, action_dy, strike,
            team_score, opposition_score,
            CASE WHEN is_home THEN team_score  ELSE opposition_score END AS home_score,
            CASE WHEN is_home THEN opposition_score ELSE team_score  END AS away_score
        FROM tracking
        WHERE has_ball
    ),

    -- Lag / lead for transition analysis
    transitions AS (
        SELECT
            bh.*,
            LAG(team_name)          OVER w AS prev_team_name,
            LAG(player_number)      OVER w AS prev_player,
            LAG(frame_number)       OVER w AS prev_frame,
            LAG(strike)             OVER w AS prev_strike,
            LAG(is_home)            OVER w AS prev_is_home,
            LAG(player_x)           OVER w AS prev_player_x,
            LAG(player_y)           OVER w AS prev_player_y,
            LAG(ball_x)             OVER w AS prev_ball_x,
            LAG(ball_y)             OVER w AS prev_ball_y,
            LAG(action_dx)          OVER w AS prev_action_dx,
            LAG(action_dy)          OVER w AS prev_action_dy,
            LAG(match_time_seconds) OVER w AS prev_match_time_seconds,
            LAG(opposition_name)    OVER w AS prev_opposition_name,
            LAG(home_score)         OVER w AS prev_home_score,
            LAG(away_score)         OVER w AS prev_away_score,
            LEAD(team_name)         OVER w AS next_team_name,
            LEAD(home_score)        OVER w AS next_home_score,
            LEAD(away_score)        OVER w AS next_away_score
        FROM ball_holders bh
        WINDOW w AS (PARTITION BY filename ORDER BY frame_number)
    ),

    -- Pre-compute y_at_goal for shot classification (only for strikes)
    strikes_with_projection AS (
        SELECT
            *,
            CASE WHEN action_dx > 1e-6
                 THEN ball_y + (action_dy / action_dx) * ({_PITCH_W} - ball_x)
                 ELSE NULL
            END AS y_at_goal
        FROM transitions
        WHERE strike
    ),

    strike_events AS (
        SELECT
            filename, frame_number,
            team_name,
            player_number,
            opposition_name,
            is_home,
            match_time_seconds,
            CASE WHEN ball_x > {_PITCH_MID_X}
                      AND action_dx > 1e-6
                      AND y_at_goal BETWEEN {_SHOT_Y_LO} AND {_SHOT_Y_HI}
                 THEN 'shot' ELSE 'pass'
            END AS event_type,
            -- success: shot → goal scored before next ball-holder frame; pass → next
            -- holder is a teammate. Both are NULL when the match ended before
            -- resolution (next_home_score / next_away_score / next_team is NULL).
            CASE WHEN ball_x > {_PITCH_MID_X}
                      AND action_dx > 1e-6
                      AND y_at_goal BETWEEN {_SHOT_Y_LO} AND {_SHOT_Y_HI}
                 THEN CASE WHEN is_home THEN (next_home_score > home_score)
                           ELSE (next_away_score > away_score)
                      END
                 ELSE CASE WHEN next_team_name IS NULL THEN NULL
                           ELSE (next_team_name = team_name)
                      END
            END AS success,
            CASE WHEN ball_x > {_PITCH_MID_X}
                      AND action_dx > 1e-6
                      AND y_at_goal BETWEEN {_SHOT_Y_LO} AND {_SHOT_Y_HI}
                 THEN y_at_goal BETWEEN {_GOAL_LO} AND {_GOAL_HI}
                 ELSE NULL
            END AS shot_on_target,
            is_human_controlled,
            player_x   AS player_x,
            player_y   AS player_y,
            ball_x     AS ball_x,
            ball_y     AS ball_y,
            action_dx,
            action_dy
        FROM strikes_with_projection
    ),

    -- Gain events: a player acquires the ball
    gain_events AS (
        SELECT
            filename, frame_number,
            team_name,
            player_number,
            opposition_name,
            is_home,
            match_time_seconds,
            CASE
                WHEN prev_frame IS NULL
                     OR home_score != prev_home_score
                     OR away_score != prev_away_score
                THEN 'kickoff'
                WHEN frame_number = prev_frame + 1
                     AND team_name != prev_team_name
                     AND NOT prev_strike
                THEN 'tackle'
                WHEN prev_strike AND team_name = prev_team_name
                THEN 'ball_receipt'
                WHEN prev_strike AND team_name != prev_team_name
                THEN 'interception'
                ELSE 'ball_receipt'
            END AS event_type,
            CAST(NULL AS BOOLEAN) AS success,
            CAST(NULL AS BOOLEAN) AS shot_on_target,
            is_human_controlled,
            player_x,
            player_y,
            ball_x,
            ball_y,
            CAST(NULL AS FLOAT)   AS action_dx,
            CAST(NULL AS FLOAT)   AS action_dy
        FROM transitions
        WHERE prev_frame IS NULL
           OR player_number != prev_player
           OR team_name != prev_team_name
           OR (prev_strike AND frame_number > prev_frame + 1)
    ),

    -- Goal / own-goal events: detected at the frame immediately before kickoff,
    -- using the full tracking data (not just ball-holder frames) so the timing
    -- is as close as possible to when the ball crossed the line.
    -- Attributed to the scoring team (goal) or conceding team (own_goal); no player.
    home_frames AS (
        -- One row per frame from the home team's perspective (avoids duplication).
        SELECT
            filename, frame_number, match_time_seconds,
            team_name        AS home_team_name,
            opposition_name  AS away_team_name,
            team_score       AS home_score,
            opposition_score AS away_score,
            LAG(frame_number)       OVER w AS prev_frame,
            LAG(match_time_seconds) OVER w AS prev_match_time_seconds,
            LAG(team_score)         OVER w AS prev_home_score,
            LAG(opposition_score)   OVER w AS prev_away_score
        FROM tracking
        WHERE is_home
        WINDOW w AS (PARTITION BY filename ORDER BY frame_number)
    ),

    goal_moments AS (
        -- Rows where the score changed since the previous frame.
        SELECT * FROM home_frames
        WHERE prev_home_score IS NOT NULL
          AND (home_score > prev_home_score OR away_score > prev_away_score)
    ),

    last_holder_at_goal AS (
        -- The last team to hold the ball at or before each goal frame,
        -- used to distinguish regular goals from own goals.
        SELECT DISTINCT ON (gm.filename, gm.prev_frame)
            gm.filename,
            gm.prev_frame      AS goal_frame,
            bh.team_name       AS last_team_name,
            bh.opposition_name AS last_opposition_name,
            bh.is_home         AS last_is_home
        FROM goal_moments gm
        JOIN ball_holders bh
            ON gm.filename = bh.filename
           AND bh.frame_number <= gm.prev_frame
        ORDER BY gm.filename, gm.prev_frame, bh.frame_number DESC
    ),

    -- One ball position row per (filename, frame, is_home perspective).
    -- All players on the same team at the same frame share the same ball coords,
    -- so MIN is just a deterministic way to collapse multiple player rows to one.
    frame_ball AS (
        SELECT filename, frame_number, is_home,
               MIN(ball_x) AS ball_x,
               MIN(ball_y) AS ball_y
        FROM tracking
        GROUP BY filename, frame_number, is_home
    ),

    goal_events AS (
        SELECT
            gm.filename,
            gm.prev_frame                        AS frame_number,
            -- goal → scoring team; own_goal → team that put it in their own net
            CASE WHEN (lh.last_is_home AND gm.home_score > gm.prev_home_score)
                   OR (NOT lh.last_is_home AND gm.away_score > gm.prev_away_score)
                 THEN lh.last_team_name
                 WHEN lh.last_is_home THEN gm.home_team_name
                 ELSE gm.away_team_name
            END                                  AS team_name,
            NULL                                 AS player_number,
            CASE WHEN (lh.last_is_home AND gm.home_score > gm.prev_home_score)
                   OR (NOT lh.last_is_home AND gm.away_score > gm.prev_away_score)
                 THEN lh.last_opposition_name
                 WHEN lh.last_is_home THEN gm.away_team_name
                 ELSE gm.home_team_name
            END                                  AS opposition_name,
            lh.last_is_home                      AS is_home,
            gm.prev_match_time_seconds           AS match_time_seconds,
            CASE WHEN (lh.last_is_home AND gm.home_score > gm.prev_home_score)
                   OR (NOT lh.last_is_home AND gm.away_score > gm.prev_away_score)
                 THEN 'goal'
                 ELSE 'own_goal'
            END                                  AS event_type,
            CAST(NULL AS BOOLEAN)                AS success,
            CAST(NULL AS BOOLEAN)                AS shot_on_target,
            CAST(NULL AS BOOLEAN)                AS is_human_controlled,
            CAST(NULL AS FLOAT)                  AS player_x,
            CAST(NULL AS FLOAT)                  AS player_y,
            -- Ball position in the attributed team's coordinate frame
            fb.ball_x,
            fb.ball_y,
            CAST(NULL AS FLOAT)                  AS action_dx,
            CAST(NULL AS FLOAT)                  AS action_dy
        FROM goal_moments gm
        LEFT JOIN last_holder_at_goal lh
            ON gm.filename = lh.filename AND gm.prev_frame = lh.goal_frame
        LEFT JOIN frame_ball fb
            ON gm.filename = fb.filename
           AND gm.prev_frame = fb.frame_number
           AND fb.is_home = lh.last_is_home
    ),

    -- Final-whistle event: one per match, attributed to the last ball-holder.
    final_whistle_events AS (
        SELECT DISTINCT ON (filename)
            filename,
            frame_number,
            team_name,
            player_number,
            opposition_name,
            is_home,
            match_time_seconds,
            'final_whistle'  AS event_type,
            CAST(NULL AS BOOLEAN) AS success,
            CAST(NULL AS BOOLEAN) AS shot_on_target,
            is_human_controlled  AS is_human_controlled,
            CAST(NULL AS FLOAT)   AS player_x,
            CAST(NULL AS FLOAT)   AS player_y,
            ball_x           AS ball_x,
            ball_y           AS ball_y,
            CAST(NULL AS FLOAT) AS action_dx,
            CAST(NULL AS FLOAT) AS action_dy
        FROM tracking
        WHERE has_ball
        ORDER BY filename, frame_number DESC
    )

    SELECT * FROM strike_events
    UNION ALL
    SELECT * FROM gain_events
    UNION ALL
    SELECT * FROM goal_events
    UNION ALL
    SELECT * FROM final_whistle_events
    """)


def _create_player_possession(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW player_possession AS
    WITH
    -- Compute the events view once so all subsequent JOINs reuse the same result.
    events_data AS (
        SELECT * FROM events
    ),

    ball_holders AS (
        SELECT
            filename, frame_number, match_time_seconds,
            team_name, player_number, is_home, opposition_name,
            ball_x, ball_y, strike
        FROM tracking
        WHERE has_ball
    ),

    with_boundary AS (
        SELECT
            bh.*,
            CASE
                WHEN LAG(frame_number) OVER w IS NULL
                     OR player_number != LAG(player_number) OVER w
                     OR team_name      != LAG(team_name)    OVER w
                     OR (LAG(strike) OVER w AND frame_number > LAG(frame_number) OVER w + 1)
                THEN 1 ELSE 0
            END AS is_new_session
        FROM ball_holders bh
        WINDOW w AS (PARTITION BY filename ORDER BY frame_number)
    ),

    with_session_id AS (
        SELECT *,
            SUM(is_new_session) OVER (
                PARTITION BY filename ORDER BY frame_number
            ) AS session_id
        FROM with_boundary
    ),

    sessions AS (
        SELECT
            filename,
            session_id,
            arg_min(team_name, frame_number)       AS team_name,
            arg_min(player_number, frame_number)   AS player_number,
            arg_min(opposition_name, frame_number) AS opposition_name,
            arg_min(is_home, frame_number)         AS is_home,
            MIN(frame_number)                      AS start_frame,
            MAX(frame_number)                      AS end_frame,
            arg_max(strike, frame_number)          AS ended_with_strike
        FROM with_session_id
        GROUP BY filename, session_id
    ),

    -- Add LEAD to get the next session's start frame, player, and team.
    -- next_team_name is used to flag whether this session is the last for its team
    -- (i.e. the next possession belongs to the opponent or there is none).
    sessions_with_lead AS (
        SELECT s.*,
            LEAD(start_frame)   OVER (PARTITION BY filename ORDER BY start_frame) AS next_start_frame,
            LEAD(player_number) OVER (PARTITION BY filename ORDER BY start_frame) AS next_player_number,
            LEAD(team_name)     OVER (PARTITION BY filename ORDER BY start_frame) AS next_team_name
        FROM sessions s
    ),

    -- Possession start event: the event at start_frame attributed to this player
    possession_start AS (
        SELECT sl.filename, sl.session_id,
               e.frame_number    AS possession_start_frame_number,
               e.match_time_seconds AS possession_start_match_time_seconds,
               e.event_type      AS possession_start_event_type,
               e.is_human_controlled AS possession_start_is_human_controlled,
               e.player_x        AS possession_start_player_x,
               e.player_y        AS possession_start_player_y,
               e.ball_x          AS possession_start_ball_x,
               e.ball_y          AS possession_start_ball_y
        FROM sessions_with_lead sl
        LEFT JOIN events_data e
            ON  sl.filename    = e.filename
            AND sl.start_frame = e.frame_number
            AND sl.player_number = e.player_number
            AND e.event_type IN ('kickoff', 'tackle', 'ball_receipt', 'interception')
    ),

    -- Possession end (direct): events at end_frame attributed to this player (pass/shot/final_whistle)
    possession_end_direct AS (
        SELECT sl.filename, sl.session_id,
               e.frame_number    AS possession_end_frame_number,
               e.match_time_seconds AS possession_end_match_time_seconds,
               e.event_type      AS possession_end_event_type,
               e.is_human_controlled AS possession_end_is_human_controlled,
               e.player_x        AS possession_end_player_x,
               e.player_y        AS possession_end_player_y,
               e.ball_x          AS possession_end_ball_x,
               e.ball_y          AS possession_end_ball_y,
               e.shot_on_target  AS possession_end_shot_on_target,
               e.success         AS possession_end_shot_success
        FROM sessions_with_lead sl
        LEFT JOIN events_data e
            ON  sl.filename      = e.filename
            AND sl.end_frame     = e.frame_number
            AND sl.player_number = e.player_number
            AND e.event_type IN ('pass', 'shot', 'final_whistle')
    ),

    -- Possession end (inferred): events at next session's start_frame attributed to the next player
    -- (tackle/interception/ball_receipt on opponent signals end of this possession)
    possession_end_inferred AS (
        SELECT sl.filename, sl.session_id,
               e.frame_number    AS possession_end_frame_number,
               e.match_time_seconds AS possession_end_match_time_seconds,
               e.event_type      AS possession_end_event_type,
               CAST(NULL AS BOOLEAN) AS possession_end_is_human_controlled,
               e.player_x        AS possession_end_player_x,
               e.player_y        AS possession_end_player_y,
               e.ball_x          AS possession_end_ball_x,
               e.ball_y          AS possession_end_ball_y
        FROM sessions_with_lead sl
        LEFT JOIN events_data e
            ON  sl.filename           = e.filename
            AND sl.next_start_frame   = e.frame_number
            AND sl.next_player_number = e.player_number
            AND e.event_type IN ('tackle', 'interception', 'ball_receipt')
    ),

    -- Combined possession end: direct takes priority over inferred
    possession_end AS (
        SELECT
            sl.filename, sl.session_id,
            COALESCE(ped.possession_end_frame_number,   pei.possession_end_frame_number)   AS possession_end_frame_number,
            COALESCE(ped.possession_end_match_time_seconds, pei.possession_end_match_time_seconds) AS possession_end_match_time_seconds,
            COALESCE(ped.possession_end_event_type,    pei.possession_end_event_type)    AS possession_end_event_type,
            COALESCE(ped.possession_end_is_human_controlled, pei.possession_end_is_human_controlled) AS possession_end_is_human_controlled,
            COALESCE(ped.possession_end_player_x,      pei.possession_end_player_x)      AS possession_end_player_x,
            COALESCE(ped.possession_end_player_y,      pei.possession_end_player_y)      AS possession_end_player_y,
            COALESCE(ped.possession_end_ball_x,        pei.possession_end_ball_x)        AS possession_end_ball_x,
            COALESCE(ped.possession_end_ball_y,        pei.possession_end_ball_y)        AS possession_end_ball_y,
            ped.possession_end_shot_on_target,
            ped.possession_end_shot_success
        FROM sessions_with_lead sl
        LEFT JOIN possession_end_direct   ped ON sl.filename = ped.filename AND sl.session_id = ped.session_id
        LEFT JOIN possession_end_inferred pei ON sl.filename = pei.filename AND sl.session_id = pei.session_id
    ),

    -- Extended possession end: for sessions that ended with a strike, find the first
    -- terminating event after the ball left (tackle/interception/ball_receipt/goal/own_goal/final_whistle).
    -- Goals/own_goals may occur at the same frame as the strike (end_frame), so we include
    -- frame_number >= end_frame for goal/own_goal but frame_number > end_frame for others.
    extended_end_frame AS (
        SELECT sl.filename, sl.session_id,
               MIN(e.frame_number) AS ext_end_frame
        FROM sessions_with_lead sl
        JOIN events_data e ON sl.filename = e.filename
        WHERE sl.ended_with_strike
          AND (
              (e.frame_number > sl.end_frame AND e.event_type IN ('tackle', 'interception', 'ball_receipt', 'final_whistle'))
              OR (e.frame_number >= sl.end_frame AND e.event_type IN ('goal', 'own_goal'))
          )
        GROUP BY sl.filename, sl.session_id
    ),

    -- Retrieve the actual event details at the extended end frame
    extended_end_event AS (
        SELECT DISTINCT ON (eef.filename, eef.session_id)
               eef.filename, eef.session_id,
               e.frame_number    AS extended_possession_end_frame_number,
               e.match_time_seconds AS extended_possession_end_match_time_seconds,
               e.event_type      AS extended_possession_end_event_type,
               e.player_x        AS extended_possession_end_player_x,
               e.player_y        AS extended_possession_end_player_y,
               e.ball_x          AS extended_possession_end_ball_x,
               e.ball_y          AS extended_possession_end_ball_y
        FROM extended_end_frame eef
        JOIN events_data e ON eef.filename = e.filename AND eef.ext_end_frame = e.frame_number
            AND e.event_type IN ('tackle', 'interception', 'ball_receipt', 'goal', 'own_goal', 'final_whistle')
        ORDER BY eef.filename, eef.session_id, e.frame_number
    )

    SELECT
        ROW_NUMBER() OVER (PARTITION BY sl.filename ORDER BY sl.start_frame) AS player_possession_id,
        sl.filename,
        sl.team_name,
        sl.player_number,
        sl.opposition_name,
        sl.is_home,
        -- Possession start
        ps.possession_start_frame_number,
        ps.possession_start_match_time_seconds,
        ps.possession_start_event_type,
        ps.possession_start_player_x,
        ps.possession_start_player_y,
        ps.possession_start_ball_x,
        ps.possession_start_ball_y,
        -- Possession end
        pe.possession_end_frame_number,
        pe.possession_end_match_time_seconds,
        pe.possession_end_event_type,
        pe.possession_end_player_x,
        pe.possession_end_player_y,
        pe.possession_end_ball_x,
        pe.possession_end_ball_y,
        pe.possession_end_match_time_seconds - ps.possession_start_match_time_seconds AS possession_duration,
        pe.possession_end_shot_on_target,
        pe.possession_end_shot_success,
        -- Extended possession end
        coalesce(ee.extended_possession_end_frame_number, pe.possession_end_frame_number) as extended_possession_end_frame_number,
        coalesce(ee.extended_possession_end_match_time_seconds, pe.possession_end_match_time_seconds) as extended_possession_end_match_time_seconds,
        coalesce(ee.extended_possession_end_event_type, pe.possession_end_event_type) as extended_possession_end_event_type,
        coalesce(ee.extended_possession_end_player_x, pe.possession_end_player_x) as extended_possession_end_player_x,
        coalesce(ee.extended_possession_end_player_y, pe.possession_end_player_y) as extended_possession_end_player_y,
        coalesce(ee.extended_possession_end_ball_x, pe.possession_end_ball_x) as extended_possession_end_ball_x,
        coalesce(ee.extended_possession_end_ball_y, pe.possession_end_ball_y) as extended_possession_end_ball_y,
        coalesce(ee.extended_possession_end_match_time_seconds, pe.possession_end_match_time_seconds) - ps.possession_start_match_time_seconds AS extended_possession_duration,
        -- True when no next possession exists or it belongs to the opponent.
        -- Used by player_match to detect failed passes without a join to team_possession.
        (sl.next_team_name IS NULL OR sl.next_team_name != sl.team_name) AS is_last_in_team_possession,
        -- True when the player was human-controlled at both the start and end of the possession.
        (ps.possession_start_is_human_controlled AND pe.possession_end_is_human_controlled) AS is_human_controlled
    FROM sessions_with_lead sl
    LEFT JOIN possession_start  ps ON sl.filename = ps.filename AND sl.session_id = ps.session_id
    LEFT JOIN possession_end    pe ON sl.filename = pe.filename AND sl.session_id = pe.session_id
    LEFT JOIN extended_end_event ee ON sl.filename = ee.filename AND sl.session_id = ee.session_id
    """)


def _create_team_possession(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW team_possession AS
    WITH
    -- Detect where team changes between consecutive player possessions
    with_boundary AS (
        SELECT *,
            CASE
                WHEN LAG(team_name) OVER (PARTITION BY filename ORDER BY possession_start_frame_number) IS NULL
                     OR team_name != LAG(team_name) OVER (PARTITION BY filename ORDER BY possession_start_frame_number)
                THEN 1 ELSE 0
            END AS is_new_session
        FROM player_possession
    ),

    with_session_id AS (
        SELECT *,
            SUM(is_new_session) OVER (
                PARTITION BY filename ORDER BY possession_start_frame_number
            ) AS team_session_id
        FROM with_boundary
    ),

    -- Identify the first and last player_possession_id within each team session
    sessions AS (
        SELECT
            filename,
            team_session_id,
            arg_min(team_name, possession_start_frame_number)       AS team_name,
            arg_min(opposition_name, possession_start_frame_number) AS opposition_name,
            arg_min(is_home, possession_start_frame_number)         AS is_home,
            arg_min(player_possession_id, possession_start_frame_number) AS first_pp_id,
            arg_max(player_possession_id, possession_start_frame_number) AS last_pp_id
        FROM with_session_id
        GROUP BY filename, team_session_id
    )

    SELECT
        ROW_NUMBER() OVER (PARTITION BY s.filename ORDER BY s.team_session_id) AS team_possession_id,
        s.filename,
        s.team_name,
        s.opposition_name,
        s.is_home,
        -- Start details from the first player possession in this team session
        pp_first.possession_start_frame_number,
        pp_first.possession_start_match_time_seconds,
        pp_first.possession_start_event_type,
        pp_first.possession_start_player_x,
        pp_first.possession_start_player_y,
        pp_first.possession_start_ball_x,
        pp_first.possession_start_ball_y,
        -- End details from the last player possession in this team session
        pp_last.possession_end_frame_number,
        pp_last.possession_end_match_time_seconds,
        pp_last.possession_end_event_type,
        pp_last.possession_end_player_x,
        pp_last.possession_end_player_y,
        pp_last.possession_end_ball_x,
        pp_last.possession_end_ball_y,
        pp_last.possession_end_match_time_seconds - pp_first.possession_start_match_time_seconds AS possession_duration,
        -- Extended end details from the last player possession in this team session
        pp_last.extended_possession_end_frame_number,
        pp_last.extended_possession_end_match_time_seconds,
        pp_last.extended_possession_end_event_type,
        pp_last.extended_possession_end_player_x,
        pp_last.extended_possession_end_player_y,
        pp_last.extended_possession_end_ball_x,
        pp_last.extended_possession_end_ball_y,
        pp_last.extended_possession_end_match_time_seconds - pp_first.possession_start_match_time_seconds AS extended_possession_duration,
    FROM sessions s
    JOIN player_possession pp_first
        ON s.filename = pp_first.filename AND s.first_pp_id = pp_first.player_possession_id
    JOIN player_possession pp_last
        ON s.filename = pp_last.filename AND s.last_pp_id = pp_last.player_possession_id
    """)


def _create_player_match(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW player_match AS
    WITH
    -- Average position: the one metric that requires the full tracking table.
    avg_position AS (
        SELECT
            filename, team_name, opposition_name, player_number, is_home,
            AVG(player_x) AS avg_x,
            AVG(player_y) AS avg_y
        FROM tracking
        GROUP BY filename, team_name, opposition_name, player_number, is_home
    ),

    -- All possession-derived stats in a single pass over player_possession.
    -- is_last_in_team_possession (pre-computed in that view) tells us whether the
    -- team lost the ball after this session, letting us infer pass success and
    -- turnovers without any additional joins.
    poss_stats AS (
        SELECT
            filename, team_name, player_number,
            COUNT(*)                                                              AS ball_touches,
            SUM(possession_end_frame_number
                - possession_start_frame_number + 1)::DOUBLE / 60.0              AS time_in_possession,
            COUNT(*) FILTER (WHERE possession_start_event_type IN ('tackle', 'interception'))
                                                                                  AS ball_recoveries,
            COUNT(*) FILTER (WHERE possession_end_event_type = 'pass')            AS passes_attempted,
            COUNT(*) FILTER (WHERE possession_end_event_type = 'pass'
                               AND NOT is_last_in_team_possession)                AS passes_successful,
            COUNT(*) FILTER (WHERE possession_end_event_type = 'shot')            AS shots_attempted,
            COUNT(*) FILTER (
                WHERE possession_end_event_type = 'shot'
                  AND possession_end_shot_on_target
                  AND possession_end_shot_success IS NOT NULL
            )                                                                     AS shots_on_target,
            COUNT(*) FILTER (WHERE extended_possession_end_event_type = 'goal')   AS goals,
            COUNT(*) FILTER (WHERE extended_possession_end_event_type = 'own_goal') AS own_goals,
            -- Turnovers: tackled directly, failed pass, or shot intercepted by opponent
            COUNT(*) FILTER (
                WHERE possession_end_event_type = 'tackle'
                   OR (possession_end_event_type = 'pass' AND is_last_in_team_possession)
                   OR (possession_end_event_type = 'shot'
                       AND extended_possession_end_event_type = 'interception')
            )                                                                     AS turnovers,
            SUM(is_last_in_team_possession::INTEGER)                              AS team_possession_count
        FROM player_possession
        GROUP BY filename, team_name, player_number
    )

    SELECT
        ap.filename,
        ap.team_name,
        ap.opposition_name,
        ap.player_number,
        ap.is_home,
        ap.avg_x,
        ap.avg_y,
        COALESCE(ps.time_in_possession, 0)  AS time_in_possession,
        COALESCE(ps.ball_touches, 0)        AS ball_touches,
        COALESCE(ps.ball_recoveries, 0)     AS ball_recoveries,
        COALESCE(ps.turnovers, 0)           AS turnovers,
        COALESCE(ps.passes_attempted, 0)    AS passes_attempted,
        COALESCE(ps.passes_successful, 0)   AS passes_successful,
        CASE WHEN COALESCE(ps.passes_attempted, 0) > 0
             THEN 100.0 * ps.passes_successful / ps.passes_attempted
             ELSE 0.0
        END                                 AS pass_completion_rate,
        COALESCE(ps.shots_attempted, 0)     AS shots_attempted,
        COALESCE(ps.shots_on_target, 0)     AS shots_on_target,
        COALESCE(ps.goals, 0)               AS goals,
        COALESCE(ps.own_goals, 0)           AS own_goals,
        COALESCE(ps.team_possession_count, 0) AS team_possession_count,
        CASE WHEN COALESCE(ps.shots_attempted, 0) > 0
             THEN 100.0 * ps.goals / ps.shots_attempted
             ELSE 0.0
        END                                 AS shot_conversion_rate
    FROM avg_position ap
    LEFT JOIN poss_stats ps ON ap.filename = ps.filename
                            AND ap.team_name = ps.team_name
                            AND ap.player_number = ps.player_number
    """)


def _create_team_match(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW team_match AS
    WITH
    team_stats AS (
        SELECT
            filename, team_name,
            ANY_VALUE(opposition_name)                              AS opposition_name,
            ANY_VALUE(is_home)                                      AS is_home,
            SUM(time_in_possession)                                 AS time_in_possession,
            SUM(ball_touches)                                       AS ball_touches,
            SUM(ball_recoveries)                                    AS ball_recoveries,
            SUM(turnovers)                                          AS turnovers,
            SUM(passes_attempted)                                   AS passes_attempted,
            SUM(passes_successful)                                  AS passes_successful,
            SUM(shots_attempted)                                    AS shots_attempted,
            SUM(shots_on_target)                                    AS shots_on_target,
            SUM(goals)                                              AS goals,
            SUM(own_goals)                                          AS own_goals
        FROM player_match
        GROUP BY filename, team_name
    )

    SELECT
        ts.filename,
        ts.team_name,
        ts.opposition_name,
        ts.is_home,
        -- Score = goals scored by this team + own goals conceded by the opponent
        ts.goals + COALESCE(opp.own_goals, 0)   AS team_score,
        opp.goals + COALESCE(ts.own_goals, 0)   AS opposition_score,
        ts.time_in_possession,
        100.0 * ts.time_in_possession
              / SUM(ts.time_in_possession) OVER (PARTITION BY ts.filename) AS possession_percentage,
        ts.ball_touches,
        ts.ball_recoveries,
        ts.turnovers,
        ts.passes_attempted,
        ts.passes_successful,
        CASE WHEN ts.passes_attempted > 0
             THEN 100.0 * ts.passes_successful / ts.passes_attempted
             ELSE 0.0
        END                                     AS pass_completion_rate,
        ts.shots_attempted,
        ts.shots_on_target,
        ts.goals,
        ts.own_goals,
        CASE WHEN ts.shots_attempted > 0
             THEN 100.0 * ts.goals / ts.shots_attempted
             ELSE 0.0
        END                                     AS shot_conversion_rate,
        strptime(
            regexp_extract(ts.filename, 'match_(\\d{8}_\\d{6})_', 1),
            '%Y%m%d_%H%M%S'
        )                                       AS timestamp
    FROM team_stats ts
    LEFT JOIN team_stats opp
        ON ts.filename = opp.filename AND ts.opposition_name = opp.team_name
    """)


def _create_human_player_match(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW human_player_match AS
    WITH
    -- Average position: filtered to frames where the player was human-controlled.
    avg_position AS (
        SELECT
            filename, team_name, opposition_name, player_number, is_home,
            AVG(player_x) AS avg_x,
            AVG(player_y) AS avg_y
        FROM tracking
        WHERE is_human_controlled
        GROUP BY filename, team_name, opposition_name, player_number, is_home
    ),

    -- Possession-derived stats, filtered to human-controlled possessions only.
    poss_stats AS (
        SELECT
            filename, team_name, player_number,
            COUNT(*)                                                              AS ball_touches,
            SUM(possession_end_frame_number
                - possession_start_frame_number + 1)::DOUBLE / 60.0              AS time_in_possession,
            COUNT(*) FILTER (WHERE possession_start_event_type IN ('tackle', 'interception'))
                                                                                  AS ball_recoveries,
            COUNT(*) FILTER (WHERE possession_end_event_type = 'pass')            AS passes_attempted,
            COUNT(*) FILTER (WHERE possession_end_event_type = 'pass'
                               AND NOT is_last_in_team_possession)                AS passes_successful,
            COUNT(*) FILTER (WHERE possession_end_event_type = 'shot')            AS shots_attempted,
            COUNT(*) FILTER (
                WHERE possession_end_event_type = 'shot'
                  AND possession_end_shot_on_target
                  AND possession_end_shot_success IS NOT NULL
            )                                                                     AS shots_on_target,
            COUNT(*) FILTER (WHERE extended_possession_end_event_type = 'goal')   AS goals,
            COUNT(*) FILTER (WHERE extended_possession_end_event_type = 'own_goal') AS own_goals,
            COUNT(*) FILTER (
                WHERE possession_end_event_type = 'tackle'
                   OR (possession_end_event_type = 'pass' AND is_last_in_team_possession)
                   OR (possession_end_event_type = 'shot'
                       AND extended_possession_end_event_type = 'interception')
            )                                                                     AS turnovers,
            SUM(is_last_in_team_possession::INTEGER)                              AS team_possession_count
        FROM player_possession
        WHERE is_human_controlled
        GROUP BY filename, team_name, player_number
    )

    SELECT
        ap.filename,
        ap.team_name,
        ap.opposition_name,
        ap.player_number,
        ap.is_home,
        ap.avg_x,
        ap.avg_y,
        COALESCE(ps.time_in_possession, 0)  AS time_in_possession,
        COALESCE(ps.ball_touches, 0)        AS ball_touches,
        COALESCE(ps.ball_recoveries, 0)     AS ball_recoveries,
        COALESCE(ps.turnovers, 0)           AS turnovers,
        COALESCE(ps.passes_attempted, 0)    AS passes_attempted,
        COALESCE(ps.passes_successful, 0)   AS passes_successful,
        CASE WHEN COALESCE(ps.passes_attempted, 0) > 0
             THEN 100.0 * ps.passes_successful / ps.passes_attempted
             ELSE 0.0
        END                                 AS pass_completion_rate,
        COALESCE(ps.shots_attempted, 0)     AS shots_attempted,
        COALESCE(ps.shots_on_target, 0)     AS shots_on_target,
        COALESCE(ps.goals, 0)               AS goals,
        COALESCE(ps.own_goals, 0)           AS own_goals,
        COALESCE(ps.team_possession_count, 0) AS team_possession_count,
        CASE WHEN COALESCE(ps.shots_attempted, 0) > 0
             THEN 100.0 * ps.goals / ps.shots_attempted
             ELSE 0.0
        END                                 AS shot_conversion_rate
    FROM avg_position ap
    LEFT JOIN poss_stats ps ON ap.filename = ps.filename
                            AND ap.team_name = ps.team_name
                            AND ap.player_number = ps.player_number
    """)


def _create_human_team_match(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW human_team_match AS
    WITH
    team_stats AS (
        SELECT
            filename, team_name,
            ANY_VALUE(opposition_name)                              AS opposition_name,
            ANY_VALUE(is_home)                                      AS is_home,
            SUM(time_in_possession)                                 AS time_in_possession,
            SUM(ball_touches)                                       AS ball_touches,
            SUM(ball_recoveries)                                    AS ball_recoveries,
            SUM(turnovers)                                          AS turnovers,
            SUM(passes_attempted)                                   AS passes_attempted,
            SUM(passes_successful)                                  AS passes_successful,
            SUM(shots_attempted)                                    AS shots_attempted,
            SUM(shots_on_target)                                    AS shots_on_target,
            SUM(goals)                                              AS goals,
            SUM(own_goals)                                          AS own_goals
        FROM human_player_match
        GROUP BY filename, team_name
    )

    SELECT
        ts.filename,
        ts.team_name,
        ts.opposition_name,
        ts.is_home,
        ts.goals + COALESCE(opp.own_goals, 0)   AS team_score,
        opp.goals + COALESCE(ts.own_goals, 0)   AS opposition_score,
        ts.time_in_possession,
        100.0 * ts.time_in_possession
              / SUM(ts.time_in_possession) OVER (PARTITION BY ts.filename) AS possession_percentage,
        ts.ball_touches,
        ts.ball_recoveries,
        ts.turnovers,
        ts.passes_attempted,
        ts.passes_successful,
        CASE WHEN ts.passes_attempted > 0
             THEN 100.0 * ts.passes_successful / ts.passes_attempted
             ELSE 0.0
        END                                     AS pass_completion_rate,
        ts.shots_attempted,
        ts.shots_on_target,
        ts.goals,
        ts.own_goals,
        CASE WHEN ts.shots_attempted > 0
             THEN 100.0 * ts.goals / ts.shots_attempted
             ELSE 0.0
        END                                     AS shot_conversion_rate,
        strptime(
            regexp_extract(ts.filename, 'match_(\\d{8}_\\d{6})_', 1),
            '%Y%m%d_%H%M%S'
        )                                       AS timestamp
    FROM team_stats ts
    LEFT JOIN team_stats opp
        ON ts.filename = opp.filename AND ts.opposition_name = opp.team_name
    """)


if __name__ == "__main__":
    build_db()
