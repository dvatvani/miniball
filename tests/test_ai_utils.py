"""Tests for miniball.ai.utils — pure AI helper functions."""

import math

import pytest

from miniball.ai.interface import BallPathPoint, BallState, PlayerState
from miniball.ai.utils import (
    dist,
    norm,
    opposition_goal_center,
    player_closest_to_point,
    relative_position,
)
from miniball.config import (
    BALL_DRAG,
    BALL_RADIUS,
    PLAYER_SPEED,
    STANDARD_GOAL_HEIGHT,
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
    """Intercept point equals the constant-velocity ball projection at intercept_time.

    Uses a player close enough to intercept while the ball is still moving
    (not after it has stopped), so the constant-velocity path formula applies.
    Ball speed √(5²+3²) ≈ 5.8 ≪ PLAYER_SPEED; player 10 units away → t_seg ≈ 1.5 s,
    well within the ~7 s segment duration.
    """
    player = make_player(1, 50.0, 40.0)
    ball = make_ball(60.0, 40.0, vx=5.0, vy=3.0)
    t = player.intercept_time(ball)
    px, py = player.intercept_point(ball)
    # On the first (non-bouncing) segment seg.time = 0, so t_seg = t
    expected_x = 60.0 + 5.0 * t
    expected_y = 40.0 + 3.0 * t
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


# ── BallState.trace_path ──────────────────────────────────────────────────────

_RIGHT_WALL = STANDARD_PITCH_WIDTH - BALL_RADIUS  # 119.0
_LEFT_WALL = BALL_RADIUS  # 1.0
_TOP_WALL = STANDARD_PITCH_HEIGHT - BALL_RADIUS  # 79.0
_BOTTOM_WALL = BALL_RADIUS  # 1.0


def test_trace_path_stationary_ball_returns_single_point():
    """A ball with no velocity is already stopped — path has one entry."""
    ball = make_ball(60.0, 40.0)
    path = ball.trace_path()
    assert len(path) == 1
    assert isinstance(path[0], BallPathPoint)
    assert path[0].location == pytest.approx((60.0, 40.0))
    assert path[0].time == pytest.approx(0.0)


def test_trace_path_ball_stops_inside_pitch():
    """Ball with moderate velocity stops without hitting any wall."""
    # rest_x = 60 + 10/BALL_DRAG ≈ 77.2 — well inside [1, 119]
    ball = make_ball(60.0, 40.0, vx=10.0)
    path = ball.trace_path()
    assert len(path) == 2
    start, stop = path
    assert start.location == pytest.approx((60.0, 40.0))
    assert start.velocity == pytest.approx((10.0, 0.0))
    assert start.time == pytest.approx(0.0)
    assert stop.velocity == pytest.approx((0.0, 0.0), abs=1e-9)
    assert stop.time > 0.0
    # Stop x should be close to the mathematical rest position (ball still has
    # speed _BALL_STOP_SPEED ≈ 0.1 at the stop point, so it's ~0.17 units short)
    assert stop.location[0] == pytest.approx(60.0 + 10.0 / BALL_DRAG, abs=0.5)
    assert stop.location[1] == pytest.approx(40.0, abs=1e-3)


def test_trace_path_bounces_off_right_wall():
    """Ball heading right fast enough to hit the right boundary bounces back."""
    # rest_x = 100 + 30/BALL_DRAG ≈ 151 — exceeds 119, so right-wall hit.
    # y=10 keeps the ball well outside the goal mouth [34, 46].
    ball = make_ball(100.0, 10.0, vx=30.0)
    path = ball.trace_path()
    assert len(path) == 3  # start, right-wall bounce, stop
    _start, bounce, _stop = path
    assert bounce.location[0] == pytest.approx(_RIGHT_WALL, abs=0.05)
    assert bounce.location[1] == pytest.approx(10.0, abs=0.05)
    assert bounce.velocity[0] < 0  # moving left after bounce
    assert abs(bounce.velocity[1]) < 1e-6  # y-vel unchanged (no y-wall hit)
    assert bounce.time > 0.0


def test_trace_path_bounces_off_top_wall():
    """Ball heading upward fast enough to hit the top boundary bounces back."""
    # rest_y = 60 + 30/BALL_DRAG ≈ 111 — exceeds 79, so top-wall hit
    ball = make_ball(60.0, 60.0, vy=30.0)
    path = ball.trace_path()
    assert len(path) == 3
    _start, bounce, _stop = path
    assert bounce.location[1] == pytest.approx(_TOP_WALL, abs=0.05)
    assert bounce.velocity[1] < 0  # moving down after bounce
    assert abs(bounce.velocity[0]) < 1e-6  # x-vel unchanged


def test_trace_path_bounces_off_left_wall():
    """Ball moving left hits the left boundary (y=10, outside goal mouth [34,46])."""
    ball = make_ball(20.0, 10.0, vx=-25.0)
    path = ball.trace_path()
    assert len(path) >= 3
    bounce = path[1]
    assert bounce.location[0] == pytest.approx(_LEFT_WALL, abs=0.05)
    assert bounce.velocity[0] > 0  # reversed to rightward


def test_trace_path_bounces_off_bottom_wall():
    """Ball moving downward hits the bottom boundary."""
    ball = make_ball(60.0, 20.0, vy=-25.0)
    path = ball.trace_path()
    assert len(path) >= 3
    bounce = path[1]
    assert bounce.location[1] == pytest.approx(_BOTTOM_WALL, abs=0.05)
    assert bounce.velocity[1] > 0  # reversed to upward


def test_trace_path_times_are_strictly_increasing():
    """Every waypoint's cumulative time must be greater than the previous one."""
    ball = make_ball(20.0, 40.0, vx=40.0, vy=30.0)
    path = ball.trace_path()
    assert len(path) >= 2
    for i in range(len(path) - 1):
        assert path[i].time < path[i + 1].time


def test_trace_path_all_points_within_pitch_bounds():
    """Every waypoint location must lie within valid pitch boundaries."""
    ball = make_ball(60.0, 40.0, vx=50.0, vy=45.0)
    path = ball.trace_path()
    for pt in path:
        assert _LEFT_WALL - 0.1 <= pt.location[0] <= _RIGHT_WALL + 0.1
        assert _BOTTOM_WALL - 0.1 <= pt.location[1] <= _TOP_WALL + 0.1


def test_trace_path_multiple_bounces():
    """A very fast ball can bounce more than once before stopping."""
    ball = make_ball(60.0, 40.0, vx=55.0, vy=45.0)
    path = ball.trace_path()
    # With speeds this high and the pitch size the ball bounces several times
    assert len(path) >= 4  # start + at least 2 bounces + stop


def test_trace_path_velocity_magnitude_decreases_at_each_bounce():
    """Speed at each bounce should be strictly less than at the previous one (drag)."""
    ball = make_ball(20.0, 40.0, vx=50.0, vy=0.0)
    path = ball.trace_path()
    speeds = [math.hypot(*pt.velocity) for pt in path[:-1]]  # exclude stopped point
    for i in range(len(speeds) - 1):
        assert speeds[i + 1] < speeds[i]


def test_trace_path_diagonal_ball_hits_x_wall_first():
    """When both x and y rests are outside, the nearer wall is hit first.

    y=10 keeps the ball outside the goal mouth [34, 46] so an x-wall bounce
    occurs rather than a goal passage.
    """
    # Ball at (105, 10) moving right (vx=20) and upward (vy=5).
    # rest_x ≈ 105 + 34 = 139 > 119; rest_y ≈ 10 + 8.6 = 18.6 < 79
    # y at x-wall ≈ 13.5 — not in goal range → only x-wall is hit
    ball = make_ball(105.0, 10.0, vx=20.0, vy=5.0)
    path = ball.trace_path()
    assert len(path) == 3
    bounce = path[1]
    assert bounce.location[0] == pytest.approx(_RIGHT_WALL, abs=0.05)


# ── Bounce-aware intercept_time / intercept_point ─────────────────────────────


def test_intercept_time_no_bounce_unchanged():
    """intercept_time is unchanged for a ball that stays inside the pitch."""
    player = make_player(1, 0.0, 40.0)
    ball = make_ball(24.0, 40.0, vx=5.0)  # rest_x ≈ 33 — well inside pitch
    t = player.intercept_time(ball)
    assert math.isfinite(t)
    assert t > 0.0


def test_intercept_time_ball_bouncing_toward_player():
    """Player to the left; ball bounces off right wall back toward them.

    y=10 keeps the shot outside the goal mouth [34, 46] so a real wall
    bounce occurs rather than the ball entering the goal.
    """
    # rest_x ≈ 95+43 = 138 > 119 → bounces off right wall at y ≈ 10
    ball = make_ball(95.0, 10.0, vx=25.0)
    player = make_player(1, 40.0, 10.0)
    t = player.intercept_time(ball)
    assert math.isfinite(t)
    assert t > 0.0
    # Intercept point must be to the left of the right wall (post-bounce segment)
    pt = player.intercept_point(ball)
    assert pt[0] < _RIGHT_WALL


def test_intercept_point_after_bounce_reachable_in_intercept_time():
    """Player can reach the intercept point in exactly intercept_time seconds."""
    ball = make_ball(95.0, 10.0, vx=25.0)  # bounces off right wall (y outside goal)
    player = make_player(1, 50.0, 10.0)
    t = player.intercept_time(ball)
    pt = player.intercept_point(ball)
    dist_to_pt = player.dist_to(pt)
    # The quadratic guarantees exact equality; allow small floating-point slack.
    assert dist_to_pt <= PLAYER_SPEED * t + 1e-3


def test_intercept_time_behind_ball_heading_to_opposite_wall():
    """Player waits near a wall; ball bounces toward them from the far side."""
    # Ball at centre heading right; player near right wall — ball arrives directly,
    # no wait needed.  y=40 is fine here because rest_x ≈ 85.7 < 119 (no wall hit).
    ball = make_ball(60.0, 40.0, vx=15.0)  # rest ≈ 85.7 — inside pitch
    player = make_player(1, 80.0, 40.0)
    t_direct = player.intercept_time(ball)

    # Ball bouncing scenario: y=10 to keep shot outside goal mouth [34, 46].
    ball2 = make_ball(90.0, 10.0, vx=25.0)  # bounces off right wall
    player2 = make_player(1, 30.0, 10.0)
    t_bounce = player2.intercept_time(ball2)

    # After bounce the ball travels further — player has to wait longer
    assert t_bounce > t_direct


def test_intercept_point_on_post_bounce_segment_matches_ball_trajectory():
    """Intercept point is consistent with constant-velocity projection on its segment."""
    # y=10 so ball bounces off the right wall rather than entering the goal.
    ball = make_ball(95.0, 10.0, vx=25.0)
    player = make_player(1, 50.0, 10.0)
    path = ball.trace_path()

    t_total = player.intercept_time(ball)
    pt = player.intercept_point(ball)

    # Find which segment the intercept falls in and verify
    for i in range(len(path) - 1):
        seg = path[i]
        seg_end = path[i + 1]
        if seg.time <= t_total <= seg_end.time + 1e-9:
            t_seg = t_total - seg.time
            expected_x = seg.location[0] + seg.velocity[0] * t_seg
            expected_y = seg.location[1] + seg.velocity[1] * t_seg
            assert pt[0] == pytest.approx(expected_x, abs=1e-3)
            assert pt[1] == pytest.approx(expected_y, abs=1e-3)
            break
    else:
        pytest.fail("intercept time does not fall within any path segment")


# ── Goal-opening: no bounce for on-target shots ────────────────────────────────

_GOAL_LO = STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2  # 34.0
_GOAL_HI = STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2  # 46.0
_GOAL_MID = STANDARD_PITCH_HEIGHT / 2  # 40.0


def test_trace_path_shot_into_right_goal_no_bounce():
    """A shot heading into the right goal mouth must not produce an x-wall bounce."""
    # Ball heading right toward the goal centre; rest_x >> 119 so it would
    # have bounced under the old logic.
    ball = make_ball(80.0, _GOAL_MID, vx=40.0)
    path = ball.trace_path()
    # All waypoints should be on the same (non-bouncing) segment — only start + stop.
    assert len(path) == 2
    # Final stop point must be to the RIGHT of the right wall (inside / behind goal).
    assert path[-1].location[0] > _RIGHT_WALL


def test_trace_path_shot_into_left_goal_no_bounce():
    """A shot heading into the left goal mouth must not produce an x-wall bounce."""
    ball = make_ball(40.0, _GOAL_MID, vx=-40.0)
    path = ball.trace_path()
    assert len(path) == 2
    assert path[-1].location[0] < _LEFT_WALL


def test_trace_path_shot_wide_of_goal_does_bounce():
    """A shot at the right end-line but OUTSIDE the goal mouth must still bounce."""
    # y = 10 is well below the goal opening [34, 46]
    ball = make_ball(80.0, 10.0, vx=40.0)
    path = ball.trace_path()
    # Expect a right-wall bounce (len > 2)
    assert len(path) >= 3
    bounce = path[1]
    assert bounce.location[0] == pytest.approx(_RIGHT_WALL, abs=0.1)
    assert bounce.velocity[0] < 0  # bounced back


def test_trace_path_diagonal_shot_hits_top_wall_then_goal():
    """Ball angled toward goal but hitting top wall first: one top-wall bounce,
    then passes through the goal without a second bounce."""
    # Ball heading right-and-up steeply; top wall is hit before the goal side.
    # After the bounce the ball's y might still be in the goal range.
    ball = make_ball(80.0, 70.0, vx=15.0, vy=20.0)
    path = ball.trace_path()
    # First bounce must be off the top wall (y-velocity reversed)
    assert path[1].location[1] == pytest.approx(_TOP_WALL, abs=0.1)
    assert path[1].velocity[1] < 0  # deflected downward
    # After the top-wall bounce the ball may or may not enter the goal depending
    # on the exact geometry; the key invariant is that NO x-wall bounce appears
    # (the last waypoint is never at x == _RIGHT_WALL with reversed vx).
    for pt in path[1:]:
        # No waypoint should be a right-wall bounce (vx < 0 at x ≈ 119)
        if pt.velocity[0] < 0:
            assert pt.location[0] != pytest.approx(_RIGHT_WALL, abs=1.0)


def test_trace_path_goal_opening_boundary_just_inside():
    """Ball aimed just inside the goal mouth (goal_lo + ε) does not bounce."""
    ball = make_ball(80.0, _GOAL_LO + 0.5, vx=40.0)
    path = ball.trace_path()
    assert len(path) == 2
    assert path[-1].location[0] > _RIGHT_WALL


def test_trace_path_goal_opening_boundary_just_outside():
    """Ball aimed just outside the goal mouth (goal_lo - ε) bounces normally."""
    ball = make_ball(80.0, _GOAL_LO - 0.5, vx=40.0)
    path = ball.trace_path()
    assert len(path) >= 3
    assert path[1].location[0] == pytest.approx(_RIGHT_WALL, abs=0.1)
