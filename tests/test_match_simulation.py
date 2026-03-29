"""Tests for MatchSimulation player-movement physics.

Focuses on the engine's responsibility to clamp ``PlayerAction`` direction
vectors at ``PLAYER_SPEED * dt`` per frame while allowing sub-speed movements
to be applied exactly as requested.
"""

import math

import pytest

from miniball.config import C_TEAM_A, C_TEAM_B, PLAYER_SPEED
from miniball.match_simulation import MatchSimulation, Player
from miniball.teams import Team


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_sim() -> MatchSimulation:
    """Minimal simulation driven by two stationary AIs."""
    return MatchSimulation(Team("A"), Team("B"))


def make_player(
    number: int = 1,
    x: float = 60.0,
    y: float = 40.0,
    is_home: bool = True,
) -> Player:
    color = C_TEAM_A if is_home else C_TEAM_B
    return Player(number=number, x=x, y=y, color=color, is_home=is_home)


# ── Movement clamping ─────────────────────────────────────────────────────────


def test_large_displacement_clamped_to_full_speed():
    """Direction whose magnitude far exceeds PLAYER_SPEED * dt moves the player
    by exactly PLAYER_SPEED * dt in that direction."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0)

    sim._apply_actions([p], {1: {"direction": (100.0, 0.0), "strike": False}}, dt)

    assert p.x == pytest.approx(60.0 + PLAYER_SPEED * dt)
    assert p.y == pytest.approx(40.0)


def test_small_displacement_applied_exactly():
    """Direction whose magnitude is below PLAYER_SPEED * dt moves the player
    by exactly that displacement — not amplified to full speed."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0)
    tiny = PLAYER_SPEED * dt * 0.1  # 10 % of max movement

    sim._apply_actions([p], {1: {"direction": (tiny, 0.0), "strike": False}}, dt)

    assert p.x == pytest.approx(60.0 + tiny)
    assert p.y == pytest.approx(40.0)


def test_displacement_at_exact_limit_applied_as_is():
    """A displacement whose magnitude equals PLAYER_SPEED * dt exactly is
    applied unchanged (boundary between the two regimes)."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0)
    max_dist = PLAYER_SPEED * dt

    sim._apply_actions([p], {1: {"direction": (max_dist, 0.0), "strike": False}}, dt)

    assert p.x == pytest.approx(60.0 + max_dist)
    assert p.y == pytest.approx(40.0)


def test_zero_direction_no_movement():
    """A zero direction vector leaves the player stationary."""
    sim = make_sim()
    p = make_player(x=60.0, y=40.0)

    sim._apply_actions([p], {1: {"direction": (0.0, 0.0), "strike": False}}, dt=1 / 60)

    assert p.x == pytest.approx(60.0)
    assert p.y == pytest.approx(40.0)


def test_clamped_diagonal_preserves_direction():
    """When clamping a diagonal displacement, the direction is unchanged and
    the resulting movement magnitude equals PLAYER_SPEED * dt."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0)

    sim._apply_actions([p], {1: {"direction": (100.0, 100.0), "strike": False}}, dt)

    actual_dx = p.x - 60.0
    actual_dy = p.y - 40.0

    assert math.hypot(actual_dx, actual_dy) == pytest.approx(PLAYER_SPEED * dt)
    assert actual_dx == pytest.approx(actual_dy)  # 45° direction preserved


def test_small_diagonal_applied_exactly():
    """A sub-speed diagonal displacement is applied without modification."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0)
    half_max = PLAYER_SPEED * dt * 0.5 / math.sqrt(2)  # keeps total < max_dist

    sim._apply_actions(
        [p], {1: {"direction": (half_max, half_max), "strike": False}}, dt
    )

    assert p.x == pytest.approx(60.0 + half_max)
    assert p.y == pytest.approx(40.0 + half_max)


def test_player_absent_from_actions_stays_still():
    """A player whose number is not in the actions dict does not move."""
    sim = make_sim()
    p = make_player(number=7, x=60.0, y=40.0)

    sim._apply_actions([p], {}, dt=1 / 60)

    assert p.x == pytest.approx(60.0)
    assert p.y == pytest.approx(40.0)


# ── Away-team coordinate flip ─────────────────────────────────────────────────


def test_away_team_direction_is_flipped():
    """An away-team player's direction is negated before being applied
    (team space attacks right → global space attacks left)."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0, is_home=False)

    # Away AI says "move right (+x) in team coords" → global = move left (−x)
    sim._apply_actions(
        [p], {1: {"direction": (100.0, 0.0), "strike": False}}, dt, is_home=False
    )

    assert p.x == pytest.approx(60.0 - PLAYER_SPEED * dt)
    assert p.y == pytest.approx(40.0)


def test_away_team_small_displacement_flipped_exactly():
    """A sub-speed away-team direction is flipped and applied without scaling."""
    sim = make_sim()
    dt = 1 / 60
    p = make_player(x=60.0, y=40.0, is_home=False)
    tiny = PLAYER_SPEED * dt * 0.1

    sim._apply_actions(
        [p], {1: {"direction": (tiny, 0.0), "strike": False}}, dt, is_home=False
    )

    assert p.x == pytest.approx(60.0 - tiny)
    assert p.y == pytest.approx(40.0)
