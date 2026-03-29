"""Tests for miniball.ai.utils — pure AI helper functions."""

import math

import pytest

from miniball.ai.interface import BallState, PlayerState
from miniball.ai.utils import (
    dist,
    goal_center,
    norm,
    player_closest_to_ball,
    player_closest_to_player,
    player_closest_to_point,
    projected_ball_position,
    projected_ball_position_when_crossing_x,
    relative_position,
)
from miniball.config import BALL_DRAG, STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_player(
    number: int,
    x: float,
    y: float,
    *,
    is_teammate: bool = True,
    has_ball: bool = False,
) -> PlayerState:
    return PlayerState(
        number=number,
        is_teammate=is_teammate,
        has_ball=has_ball,
        cooldown_timer=0.0,
        location=(x, y),
    )


def make_ball(x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> BallState:
    return BallState(location=(x, y), velocity=(vx, vy))


# ── dist ──────────────────────────────────────────────────────────────────────


def test_dist_3_4_5_triangle():
    assert dist([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_dist_same_point():
    assert dist([7.0, 3.0], [7.0, 3.0]) == pytest.approx(0.0)


def test_dist_negative_coords():
    assert dist([0.0, 0.0], [-3.0, -4.0]) == pytest.approx(5.0)


def test_dist_is_symmetric():
    a, b = [10.0, 20.0], [30.0, 50.0]
    assert dist(a, b) == pytest.approx(dist(b, a))


# ── norm ──────────────────────────────────────────────────────────────────────


def test_norm_produces_unit_vector():
    dx, dy = norm(3.0, 4.0)
    assert math.hypot(dx, dy) == pytest.approx(1.0)
    assert dx == pytest.approx(0.6)
    assert dy == pytest.approx(0.8)


def test_norm_already_unit():
    dx, dy = norm(1.0, 0.0)
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(0.0)


def test_norm_negative_direction():
    dx, dy = norm(-1.0, 0.0)
    assert dx == pytest.approx(-1.0)
    assert dy == pytest.approx(0.0)


def test_norm_zero_vector_returns_zeros():
    assert norm(0.0, 0.0) == (0.0, 0.0)


def test_norm_near_zero_vector_returns_zeros():
    assert norm(1e-8, 1e-8) == (0.0, 0.0)


# ── relative_position ─────────────────────────────────────────────────────────


def test_relative_position_returns_tuple():
    result = relative_position([1.0, 2.0], [4.0, 6.0])
    assert isinstance(result, tuple)
    assert result == pytest.approx((3.0, 4.0))


def test_relative_position_same_point_is_zero():
    assert relative_position([5.0, 3.0], [5.0, 3.0]) == pytest.approx((0.0, 0.0))


def test_relative_position_can_be_negative():
    assert relative_position([10.0, 10.0], [5.0, 3.0]) == pytest.approx((-5.0, -7.0))


def test_relative_position_accepts_tuple_inputs():
    """Inputs can be tuples (e.g. the output of goal_center())."""
    assert relative_position((0.0, 0.0), (3.0, 4.0)) == pytest.approx((3.0, 4.0))


def test_dist_accepts_tuple_inputs():
    """dist should accept tuples so callers can pass goal_center() directly."""
    assert dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# ── goal_center ───────────────────────────────────────────────────────────────


def test_goal_center_is_right_edge_mid_height():
    gx, gy = goal_center()
    assert gx == STANDARD_PITCH_WIDTH
    assert gy == pytest.approx(STANDARD_PITCH_HEIGHT / 2)


# ── player_closest_to_point ───────────────────────────────────────────────────


def test_player_closest_to_point_returns_nearest():
    players = [
        make_player(1, 10.0, 10.0),
        make_player(2, 50.0, 40.0),
        make_player(3, 90.0, 70.0),
    ]
    result = player_closest_to_point(players, [52.0, 42.0])
    assert result.number == 2


def test_player_closest_to_point_single_player():
    players = [make_player(1, 60.0, 40.0)]
    result = player_closest_to_point(players, [0.0, 0.0])
    assert result.number == 1


def test_player_closest_to_point_tie_broken_by_list_order():
    """When two players are equidistant, the one listed first is returned."""
    players = [
        make_player(1, 0.0, 5.0),
        make_player(2, 0.0, -5.0),
    ]
    result = player_closest_to_point(players, [0.0, 0.0])
    assert result.number in (1, 2)


# ── player_closest_to_ball ────────────────────────────────────────────────────


def test_player_closest_to_ball_uses_ball_location():
    ball = make_ball(60.0, 40.0)
    players = [
        make_player(1, 20.0, 20.0),
        make_player(2, 61.0, 40.0),
    ]
    result = player_closest_to_ball(players, ball)
    assert result.number == 2


# ── player_closest_to_player ─────────────────────────────────────────────────


def test_player_closest_to_player_ignores_self_by_default():
    """With ignore_self=True (default), the player's own entry is excluded."""
    players = [
        make_player(1, 60.0, 40.0),  # the reference player
        make_player(2, 61.0, 40.0),  # closest other teammate
        make_player(3, 90.0, 70.0),
    ]
    result = player_closest_to_player(players[0], players)
    assert result.number == 2


def test_player_closest_to_player_include_self():
    """With ignore_self=False, the player can be returned as their own nearest."""
    players = [
        make_player(1, 60.0, 40.0),
        make_player(2, 90.0, 70.0),
    ]
    result = player_closest_to_player(players[0], players, ignore_self=False)
    assert result.number == 1


def test_player_closest_to_player_includes_opponents_when_in_list():
    """Opponents in the list are not filtered out — caller controls the pool."""
    ref = make_player(1, 60.0, 40.0, is_teammate=True)
    opponent_nearby = make_player(2, 61.0, 40.0, is_teammate=False)
    teammate_far = make_player(3, 90.0, 70.0, is_teammate=True)
    result = player_closest_to_player(ref, [ref, opponent_nearby, teammate_far])
    assert result.number == 2  # nearest player regardless of team


def test_player_closest_to_player_teammates_only_when_filtered():
    """Passing only teammates preserves the old team-scoped behaviour."""
    ref = make_player(1, 60.0, 40.0, is_teammate=True)
    opponent_nearby = make_player(2, 61.0, 40.0, is_teammate=False)
    teammate_far = make_player(3, 90.0, 70.0, is_teammate=True)
    teammates = [p for p in [ref, opponent_nearby, teammate_far] if p.is_teammate]
    result = player_closest_to_player(ref, teammates)
    assert (
        result.number == 3
    )  # opponent excluded by caller; teammate_far is only option


# ── projected_ball_position ───────────────────────────────────────────────────


def test_projected_ball_position_zero_time_returns_current():
    ball = make_ball(60.0, 40.0, vx=20.0, vy=10.0)
    x, y = projected_ball_position(ball, 0.0)
    assert x == pytest.approx(60.0)
    assert y == pytest.approx(40.0)


def test_projected_ball_position_negative_time_returns_current():
    ball = make_ball(60.0, 40.0, vx=20.0, vy=10.0)
    x, y = projected_ball_position(ball, -1.0)
    assert x == pytest.approx(60.0)
    assert y == pytest.approx(40.0)


def test_projected_ball_position_no_velocity_stays_put():
    ball = make_ball(30.0, 50.0, vx=0.0, vy=0.0)
    x, y = projected_ball_position(ball, 2.0)
    assert x == pytest.approx(30.0)
    assert y == pytest.approx(50.0)


def test_projected_ball_position_matches_drag_formula():
    """Verify against the continuous-drag formula directly."""
    ball = make_ball(60.0, 40.0, vx=10.0, vy=0.0)
    t = 1.0
    decay = math.exp(-BALL_DRAG * t)
    factor = (1.0 - decay) / BALL_DRAG
    expected_x = 60.0 + 10.0 * factor
    x, y = projected_ball_position(ball, t)
    assert x == pytest.approx(expected_x, rel=1e-6)
    assert y == pytest.approx(40.0)


def test_projected_ball_position_ball_decelerates():
    """Projected position with drag must be closer to start than without drag."""
    ball = make_ball(0.0, 0.0, vx=10.0, vy=0.0)
    t = 2.0
    x_drag, _ = projected_ball_position(ball, t)
    x_no_drag = 0.0 + 10.0 * t  # constant velocity (no drag)
    assert x_drag < x_no_drag


def test_projected_ball_position_2d():
    ball = make_ball(50.0, 30.0, vx=6.0, vy=8.0)
    t = 0.5
    decay = math.exp(-BALL_DRAG * t)
    factor = (1.0 - decay) / BALL_DRAG
    x, y = projected_ball_position(ball, t)
    assert x == pytest.approx(50.0 + 6.0 * factor, rel=1e-6)
    assert y == pytest.approx(30.0 + 8.0 * factor, rel=1e-6)


# ── projected_ball_position_when_crossing_x ───────────────────────────────────


def test_crossing_x_stationary_ball_returns_none():
    ball = make_ball(60.0, 40.0, vx=0.0, vy=0.0)
    assert projected_ball_position_when_crossing_x(ball, 80.0) is None


def test_crossing_x_ball_already_at_target_returns_none():
    ball = make_ball(60.0, 40.0, vx=10.0, vy=0.0)
    assert projected_ball_position_when_crossing_x(ball, 60.0) is None


def test_crossing_x_returns_tuple():
    ball = make_ball(60.0, 40.0, vx=10.0, vy=0.0)
    result = projected_ball_position_when_crossing_x(ball, 80.0)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_crossing_x_forward_moving_ball():
    """Ball moving right should end up near the target x."""
    ball = make_ball(60.0, 40.0, vx=20.0, vy=0.0)
    result = projected_ball_position_when_crossing_x(ball, 80.0)
    assert result is not None
    # Due to drag the actual projected x won't be exactly 80, but should be close
    x, y = result
    assert 60.0 < x
    assert y == pytest.approx(40.0)


def test_crossing_x_with_lateral_velocity():
    """A ball with both vx and vy should deflect in y at the crossing point."""
    ball = make_ball(60.0, 40.0, vx=10.0, vy=5.0)
    result = projected_ball_position_when_crossing_x(ball, 80.0)
    assert result is not None
    _, y = result
    assert y > 40.0  # ball drifts upward
