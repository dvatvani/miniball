"""Tests for GameState, focusing on the players() filtering method."""

import pytest

from miniball.ai.interface import BallState, GameState, MatchState, PlayerState
from miniball.config import STANDARD_PITCH_HEIGHT

# ── Helpers ───────────────────────────────────────────────────────────────────

_MID_Y = STANDARD_PITCH_HEIGHT / 2


def make_player(
    number: int,
    x: float,
    y: float = _MID_Y,
    *,
    is_teammate: bool = True,
    is_home: bool = True,
    has_ball: bool = False,
    cooldown_timer: float = 0.0,
) -> PlayerState:
    return PlayerState(
        number=number,
        is_teammate=is_teammate,
        is_home=is_home,
        has_ball=has_ball,
        cooldown_timer=cooldown_timer,
        location=(x, y),
    )


def make_state(
    team: list[PlayerState],
    opposition: list[PlayerState],
    *,
    is_home: bool = True,
) -> GameState:
    return GameState(
        team=team,
        opposition=opposition,
        ball=BallState(location=(60.0, _MID_Y), velocity=(0.0, 0.0)),
        match_state=MatchState(
            team_current_score=0,
            opposition_current_score=0,
            match_time_seconds=0.0,
        ),
        is_home=is_home,
    )


# Standard 5-a-side setup used across most tests.
# Team: GK near own goal (x≈2), 4 outfield players spread across pitch.
# Opposition: mirror image (their GK near x≈118, their own goal at x=120).


def _standard_state() -> GameState:
    team = [
        make_player(1, 2.0),  # GK – closest to own goal (x=0)
        make_player(2, 30.0),
        make_player(3, 50.0),
        make_player(4, 70.0),
        make_player(5, 90.0),
    ]
    opposition = [
        make_player(
            1, 118.0, is_teammate=False, is_home=False
        ),  # opp GK – closest to opp goal (x=120)
        make_player(2, 90.0, is_teammate=False, is_home=False),
        make_player(3, 70.0, is_teammate=False, is_home=False),
        make_player(4, 50.0, is_teammate=False, is_home=False),
        make_player(5, 30.0, is_teammate=False, is_home=False),
    ]
    return make_state(team, opposition)


# ── players() — default (no filters) ─────────────────────────────────────────


def test_players_default_returns_all():
    state = _standard_state()
    result = state.players()
    assert len(result) == 10
    assert set(result) == set(state.all_players)


# ── teammates / opposition toggles ───────────────────────────────────────────


def test_players_teammates_only():
    state = _standard_state()
    result = state.players(opposition=False)
    assert result == state.team


def test_players_opposition_only():
    state = _standard_state()
    result = state.players(teammates=False)
    assert result == state.opposition


def test_players_neither_team_returns_empty():
    state = _standard_state()
    assert state.players(teammates=False, opposition=False) == []


# ── Goalkeeper identification ─────────────────────────────────────────────────


def test_team_gk_is_player_closest_to_own_goal():
    """Player closest to x=0 should be identified as the team GK."""
    state = _standard_state()
    outfield = state.players(include_goalkeepers=False)
    # GK (number=1, x=2) must be absent; all others present
    team_outfield = [p for p in outfield if p.is_teammate]
    assert all(p.number != 1 for p in team_outfield)
    assert len(team_outfield) == 4


def test_opposition_gk_is_player_closest_to_their_goal():
    """Opposition player closest to x=120 should be identified as the opp GK."""
    state = _standard_state()
    outfield = state.players(include_goalkeepers=False)
    opp_outfield = [p for p in outfield if not p.is_teammate]
    assert all(p.number != 1 for p in opp_outfield)
    assert len(opp_outfield) == 4


def test_include_outfield_false_returns_only_goalkeepers():
    state = _standard_state()
    gks = state.players(include_outfield=False)
    assert len(gks) == 2
    # Each team contributes exactly 1 GK
    team_gks = [p for p in gks if p.is_teammate]
    opp_gks = [p for p in gks if not p.is_teammate]
    assert len(team_gks) == 1
    assert len(opp_gks) == 1
    assert team_gks[0].number == 1  # closest to x=0
    assert opp_gks[0].number == 1  # closest to x=120


def test_include_neither_returns_empty():
    state = _standard_state()
    assert state.players(include_goalkeepers=False, include_outfield=False) == []


# ── n_goalkeepers ─────────────────────────────────────────────────────────────


def test_two_goalkeepers_per_team():
    state = _standard_state()
    gks = state.players(include_outfield=False, n_goalkeepers=2)
    assert len(gks) == 4  # 2 per team

    team_gks = sorted([p for p in gks if p.is_teammate], key=lambda p: p.location[0])
    assert team_gks[0].location[0] == pytest.approx(2.0)  # x=2 (closest to own goal)
    assert team_gks[1].location[0] == pytest.approx(30.0)  # x=30 (second closest)


def test_n_goalkeepers_zero_means_all_outfield():
    """With n_goalkeepers=0 there are no GKs, so include_goalkeepers=False excludes nobody."""
    state = _standard_state()
    result = state.players(include_goalkeepers=False, n_goalkeepers=0)
    assert len(result) == 10


def test_n_goalkeepers_exceeds_team_size_clamps_gracefully():
    """Requesting more GKs than players shouldn't raise; all players become GKs."""
    state = _standard_state()
    gks = state.players(include_outfield=False, n_goalkeepers=10)
    assert len(gks) == 10  # every player is a GK


# ── GK identification is proximity-based, not number-based ───────────────────


def test_gk_identified_by_proximity_not_number():
    """Player #5 positioned closest to own goal becomes GK, not player #1."""
    team = [
        make_player(1, 90.0),  # far from own goal
        make_player(2, 70.0),
        make_player(3, 50.0),
        make_player(4, 30.0),
        make_player(5, 2.0),  # closest to own goal → GK
    ]
    opposition = [
        make_player(1, 118.0, is_teammate=False, is_home=False),
        make_player(2, 90.0, is_teammate=False, is_home=False),
        make_player(3, 70.0, is_teammate=False, is_home=False),
        make_player(4, 50.0, is_teammate=False, is_home=False),
        make_player(5, 30.0, is_teammate=False, is_home=False),
    ]
    state = make_state(team, opposition)
    team_gks = state.players(include_outfield=False, teammates=True, opposition=False)
    assert len(team_gks) == 1
    assert team_gks[0].number == 5


# ── Combining goalkeeper filters with teammates/opposition toggles ────────────


def test_team_outfield_only():
    state = _standard_state()
    result = state.players(opposition=False, include_goalkeepers=False)
    assert len(result) == 4
    assert all(p.is_teammate for p in result)
    assert all(p.number != 1 for p in result)


def test_opposition_goalkeepers_only():
    state = _standard_state()
    result = state.players(teammates=False, include_outfield=False)
    assert len(result) == 1
    assert not result[0].is_teammate
    assert result[0].number == 1


# ── Combining with player_on_ball and players_on_cooldown ────────────────────


def test_player_on_ball_false_excludes_gk_with_ball():
    team = [
        make_player(1, 2.0, has_ball=True),  # GK has the ball
        make_player(2, 30.0),
        make_player(3, 50.0),
        make_player(4, 70.0),
        make_player(5, 90.0),
    ]
    opposition = [
        make_player(1, 118.0, is_teammate=False, is_home=False),
        make_player(2, 90.0, is_teammate=False, is_home=False),
        make_player(3, 70.0, is_teammate=False, is_home=False),
        make_player(4, 50.0, is_teammate=False, is_home=False),
        make_player(5, 30.0, is_teammate=False, is_home=False),
    ]
    state = make_state(team, opposition)
    # GK has ball → excluded by player_on_ball=False, regardless of GK status
    result = state.players(player_on_ball=False, include_outfield=False)
    assert all(p.number != 1 or not p.is_teammate for p in result)


def test_players_on_cooldown_false_excludes_cooldown_player():
    team = [
        make_player(1, 2.0),
        make_player(2, 30.0, cooldown_timer=0.5),  # on cooldown
        make_player(3, 50.0),
        make_player(4, 70.0),
        make_player(5, 90.0),
    ]
    opposition = [
        make_player(1, 118.0, is_teammate=False, is_home=False),
        make_player(2, 90.0, is_teammate=False, is_home=False),
        make_player(3, 70.0, is_teammate=False, is_home=False),
        make_player(4, 50.0, is_teammate=False, is_home=False),
        make_player(5, 30.0, is_teammate=False, is_home=False),
    ]
    state = make_state(team, opposition)
    result = state.players(opposition=False, players_on_cooldown=False)
    assert len(result) == 4
    assert all(p.number != 2 for p in result)
