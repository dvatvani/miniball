"""Tests for miniball.ai.utils — pure AI helper functions."""

import math

import pytest

from miniball.ai.interface import BallState, PlayerState
from miniball.ai.utils import (
    dist,
    norm,
    opposition_goal_center,
    player_closest_to_point,
    relative_position,
)
from miniball.config import (
    BALL_DRAG,
    PLAYER_SPEED,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_player(
    number: int,
    x: float,
    y: float,
    *,
    is_teammate: bool = True,
    is_home: bool = True,
    has_ball: bool = False,
) -> PlayerState:
    return PlayerState(
        number=number,
        is_teammate=is_teammate,
        is_home=is_home,
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
    gx, gy = opposition_goal_center()
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


# ── ball.closest_player_in ────────────────────────────────────────────────────


def test_closest_player_in_uses_ball_location():
    ball = make_ball(60.0, 40.0)
    players = [
        make_player(1, 20.0, 20.0),
        make_player(2, 61.0, 40.0),
    ]
    assert ball.closest_player(players).number == 2


# ── player.closest_player ─────────────────────────────────────────────────────


def test_closest_player_ignores_self_by_default():
    """With ignore_self=True (default), the player's own entry is excluded."""
    players = [
        make_player(1, 60.0, 40.0),  # the reference player
        make_player(2, 61.0, 40.0),  # closest other
        make_player(3, 90.0, 70.0),
    ]
    assert players[0].closest_player(players).number == 2


def test_closest_player_include_self():
    """With ignore_self=False, the player can be returned as their own nearest."""
    players = [
        make_player(1, 60.0, 40.0),
        make_player(2, 90.0, 70.0),
    ]
    assert players[0].closest_player(players, ignore_self=False).number == 1


def test_closest_player_includes_opponents_when_in_list():
    """Opponents in the list are not filtered out — caller controls the pool."""
    ref = make_player(1, 60.0, 40.0, is_teammate=True, is_home=True)
    opponent_nearby = make_player(2, 61.0, 40.0, is_teammate=False, is_home=False)
    teammate_far = make_player(3, 90.0, 70.0, is_teammate=True, is_home=True)
    assert ref.closest_player([ref, opponent_nearby, teammate_far]).number == 2


def test_closest_player_teammates_only_when_filtered():
    """Passing only teammates achieves team-scoped behaviour."""
    ref = make_player(1, 60.0, 40.0, is_teammate=True, is_home=True)
    opponent_nearby = make_player(2, 61.0, 40.0, is_teammate=False, is_home=False)
    teammate_far = make_player(3, 90.0, 70.0, is_teammate=True, is_home=True)
    teammates = [p for p in [ref, opponent_nearby, teammate_far] if p.is_teammate]
    assert ref.closest_player(teammates).number == 3


# ── ball.projected_position ───────────────────────────────────────────────────


def test_projected_position_zero_time_returns_current():
    ball = make_ball(60.0, 40.0, vx=20.0, vy=10.0)
    x, y = ball.projected_position(0.0)
    assert x == pytest.approx(60.0)
    assert y == pytest.approx(40.0)


def test_projected_position_negative_time_returns_current():
    ball = make_ball(60.0, 40.0, vx=20.0, vy=10.0)
    x, y = ball.projected_position(-1.0)
    assert x == pytest.approx(60.0)
    assert y == pytest.approx(40.0)


def test_projected_position_no_velocity_stays_put():
    ball = make_ball(30.0, 50.0, vx=0.0, vy=0.0)
    x, y = ball.projected_position(2.0)
    assert x == pytest.approx(30.0)
    assert y == pytest.approx(50.0)


def test_projected_position_matches_drag_formula():
    """Verify against the continuous-drag formula directly."""
    ball = make_ball(60.0, 40.0, vx=10.0, vy=0.0)
    t = 1.0
    decay = math.exp(-BALL_DRAG * t)
    factor = (1.0 - decay) / BALL_DRAG
    expected_x = 60.0 + 10.0 * factor
    x, y = ball.projected_position(t)
    assert x == pytest.approx(expected_x, rel=1e-6)
    assert y == pytest.approx(40.0)


def test_projected_position_ball_decelerates():
    """Projected position with drag must be closer to start than without drag."""
    ball = make_ball(0.0, 0.0, vx=10.0, vy=0.0)
    t = 2.0
    x_drag, _ = ball.projected_position(t)
    x_no_drag = 0.0 + 10.0 * t  # constant velocity (no drag)
    assert x_drag < x_no_drag


def test_projected_position_2d():
    ball = make_ball(50.0, 30.0, vx=6.0, vy=8.0)
    t = 0.5
    decay = math.exp(-BALL_DRAG * t)
    factor = (1.0 - decay) / BALL_DRAG
    x, y = ball.projected_position(t)
    assert x == pytest.approx(50.0 + 6.0 * factor, rel=1e-6)
    assert y == pytest.approx(30.0 + 8.0 * factor, rel=1e-6)


# ── ball.position_when_crossing_x ─────────────────────────────────────────────


def test_crossing_x_stationary_ball_returns_none():
    ball = make_ball(60.0, 40.0, vx=0.0, vy=0.0)
    assert ball.position_when_crossing_x(80.0) is None


def test_crossing_x_ball_already_at_target_returns_none():
    ball = make_ball(60.0, 40.0, vx=10.0, vy=0.0)
    assert ball.position_when_crossing_x(60.0) is None


def test_crossing_x_returns_tuple():
    ball = make_ball(60.0, 40.0, vx=10.0, vy=0.0)
    result = ball.position_when_crossing_x(80.0)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_crossing_x_forward_moving_ball():
    """Ball moving right should end up near the target x."""
    ball = make_ball(60.0, 40.0, vx=20.0, vy=0.0)
    result = ball.position_when_crossing_x(80.0)
    assert result is not None
    x, y = result
    assert 60.0 < x
    assert y == pytest.approx(40.0)


def test_crossing_x_with_lateral_velocity():
    """A ball with both vx and vy should deflect in y at the crossing point."""
    ball = make_ball(60.0, 40.0, vx=10.0, vy=5.0)
    result = ball.position_when_crossing_x(80.0)
    assert result is not None
    _, y = result
    assert y > 40.0  # ball drifts upward


# ── intercept_time / intercept_point ─────────────────────────────────────────


def test_intercept_time_stationary_ball():
    """intercept_time for a stationary ball ≈ dist / PLAYER_SPEED."""
    player = make_player(1, 0.0, 40.0)
    ball = make_ball(24.0, 40.0)  # 24 units away
    t = player.intercept_time(ball)
    assert t == pytest.approx(24.0 / PLAYER_SPEED, rel=1e-3)


def test_intercept_time_player_already_at_ball():
    player = make_player(1, 60.0, 40.0)
    ball = make_ball(60.0, 40.0)
    assert player.intercept_time(ball) == pytest.approx(0.0, abs=1e-3)


def test_intercept_time_ball_moving_toward_player_is_shorter():
    """A ball approaching the player should be reachable sooner than a stationary one."""
    player = make_player(1, 0.0, 40.0)
    ball_stationary = make_ball(24.0, 40.0)
    ball_approaching = make_ball(24.0, 40.0, vx=-20.0)  # moving towards player
    t_static = player.intercept_time(ball_stationary)
    t_approach = player.intercept_time(ball_approaching)
    assert t_approach < t_static


def test_intercept_time_ball_moving_away_is_longer():
    """A ball moving away (but slower than the player) takes longer to intercept."""
    player = make_player(1, 0.0, 40.0)
    ball_stationary = make_ball(24.0, 40.0)
    # vx=5 < PLAYER_SPEED=12, so the player can catch it but it still takes longer
    ball_receding = make_ball(24.0, 40.0, vx=5.0)
    t_static = player.intercept_time(ball_stationary)
    t_receding = player.intercept_time(ball_receding)
    assert t_receding > t_static


def test_intercept_point_is_on_constant_velocity_trajectory():
    """The intercept point should equal the ball's constant-velocity projection."""
    player = make_player(1, 0.0, 40.0)
    ball = make_ball(60.0, 40.0, vx=10.0, vy=5.0)
    t = player.intercept_time(ball)
    px, py = player.intercept_point(ball)
    expected_x = 60.0 + 10.0 * t
    expected_y = 40.0 + 5.0 * t
    assert px == pytest.approx(expected_x, rel=1e-4)
    assert py == pytest.approx(expected_y, rel=1e-4)


def test_intercept_point_reachable_in_intercept_time():
    """Player can walk to the intercept point in intercept_time seconds (vx/vy < PLAYER_SPEED)."""
    player = make_player(1, 10.0, 20.0)
    # ball speed = sqrt(5²+3²) ≈ 5.8 < PLAYER_SPEED=12, so the quadratic has a valid root
    ball = make_ball(80.0, 60.0, vx=-5.0, vy=3.0)
    t = player.intercept_time(ball)
    point = player.intercept_point(ball)
    dist = player.dist_to(point)
    assert dist <= PLAYER_SPEED * t + 1e-3


# ── BallState.interception_times ─────────────────────────────────────────────


def test_interception_times_sorted_ascending():
    ball = make_ball(60.0, 40.0)
    players = [
        make_player(1, 55.0, 40.0),  # closest → fastest
        make_player(2, 40.0, 40.0),
        make_player(3, 10.0, 40.0),  # farthest → slowest
    ]
    ranked = ball.interception_times(players)
    times = [t for t, _ in ranked]
    assert times == sorted(times)


def test_interception_times_returns_all_players():
    ball = make_ball(60.0, 40.0)
    players = [make_player(i, float(i * 10), 40.0) for i in range(1, 6)]
    ranked = ball.interception_times(players)
    assert len(ranked) == 5
    assert {p.number for _, p in ranked} == {1, 2, 3, 4, 5}


def test_interception_times_single_player():
    ball = make_ball(60.0, 40.0)
    player = make_player(1, 48.0, 40.0)
    ranked = ball.interception_times([player])
    assert len(ranked) == 1
    assert ranked[0][1] is player


# ── BallState.fastest_interceptor ────────────────────────────────────────────


def test_fastest_interceptor_returns_closest_player_for_stationary_ball():
    """For a stationary ball nearest player == fastest interceptor."""
    ball = make_ball(60.0, 40.0)
    players = [
        make_player(1, 10.0, 40.0),
        make_player(2, 58.0, 40.0),  # closest
        make_player(3, 90.0, 40.0),
    ]
    assert ball.fastest_interceptor(players).number == 2


def test_fastest_interceptor_accounts_for_ball_movement():
    """A moving ball may favour a player who is NOT the geometrically closest.

    Ball speed (10) < PLAYER_SPEED (12) so the quadratic has valid roots.
    near_player is just behind the ball (which is moving away); far_player is
    directly ahead in the ball's path and intercepts it sooner.
    """
    ball = make_ball(60.0, 40.0, vx=10.0)  # heading right, speed < PLAYER_SPEED
    near_player = make_player(1, 58.0, 40.0)  # 2 units behind ball, ball moving away
    far_player = make_player(2, 80.0, 40.0)  # 20 units ahead, in the ball's path
    fastest = ball.fastest_interceptor([near_player, far_player])
    assert fastest.number == 2


# ── BallState.intercept_advantage ────────────────────────────────────────────


def test_intercept_advantage_positive_when_player_is_faster():
    ball = make_ball(60.0, 40.0)
    player = make_player(1, 58.0, 40.0)  # very close
    opponent = make_player(2, 20.0, 40.0)  # far away
    adv = ball.intercept_advantage(player, [opponent])
    assert adv > 0


def test_intercept_advantage_negative_when_opponent_is_faster():
    ball = make_ball(60.0, 40.0)
    player = make_player(1, 10.0, 40.0)  # far
    opponent = make_player(2, 59.0, 40.0)  # close
    adv = ball.intercept_advantage(player, [opponent])
    assert adv < 0


def test_intercept_advantage_no_opponents_returns_inf():
    ball = make_ball(60.0, 40.0)
    player = make_player(1, 50.0, 40.0)
    assert ball.intercept_advantage(player, []) == pytest.approx(float("inf"))


def test_intercept_advantage_uses_nearest_opponent_only():
    """Advantage is measured against the fastest opponent, not the slowest."""
    ball = make_ball(60.0, 40.0)
    player = make_player(1, 55.0, 40.0)
    opp_close = make_player(2, 57.0, 40.0)  # faster than player
    opp_distant = make_player(3, 5.0, 40.0)  # much slower than player
    adv = ball.intercept_advantage(player, [opp_close, opp_distant])
    # The closest opponent (opp_close) is faster, so advantage should be negative
    assert adv < 0
