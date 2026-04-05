"""Tests for the three-coordinate-system conversion functions."""

from miniball.coords import (
    global_delta_to_team,
    global_to_team,
    team_delta_to_global,
    team_to_global,
)
from miniball.ui.config import PITCH_B, PITCH_L, PITCH_R, PITCH_T
from miniball.ui.coords import (
    global_delta_to_screen,
    global_to_screen,
    screen_delta_to_global,
    screen_delta_to_team,
    screen_to_global,
    screen_to_team,
    team_delta_to_screen,
    team_to_screen,
)

# ── Position: Screen ↔ Global ─────────────────────────────────────────────────


def test_screen_to_global_pitch_corners():
    """Pitch corners map to standard pitch corners."""
    assert screen_to_global(PITCH_L, PITCH_B) == (0.0, 0.0)
    assert screen_to_global(PITCH_R, PITCH_T) == (120.0, 80.0)


def test_screen_to_global_pitch_centre():
    cx = (PITCH_L + PITCH_R) / 2
    cy = (PITCH_B + PITCH_T) / 2
    assert screen_to_global(cx, cy) == (60.0, 40.0)


def test_screen_global_round_trip():
    for sx, sy in [(PITCH_L, PITCH_B), (PITCH_R, PITCH_T), (600, 400)]:
        gx, gy = screen_to_global(sx, sy)
        sx2, sy2 = global_to_screen(gx, gy)
        assert abs(sx2 - sx) < 1e-9
        assert abs(sy2 - sy) < 1e-9


# ── Position: Global ↔ Team ───────────────────────────────────────────────────


def test_global_to_team_home_is_identity():
    assert global_to_team(30.0, 25.0, is_home=True) == (30.0, 25.0)


def test_global_to_team_away_is_rotation():
    assert global_to_team(30.0, 20.0, is_home=False) == (90.0, 60.0)


def test_global_team_round_trip():
    for is_home in (True, False):
        for gx, gy in [(0, 0), (60, 40), (120, 80), (10, 70)]:
            tx, ty = global_to_team(gx, gy, is_home=is_home)
            gx2, gy2 = team_to_global(tx, ty, is_home=is_home)
            assert abs(gx2 - gx) < 1e-9
            assert abs(gy2 - gy) < 1e-9


# ── Position: Screen ↔ Team ───────────────────────────────────────────────────


def test_screen_to_team_home_same_as_global():
    sx, sy = 700.0, 400.0
    assert screen_to_team(sx, sy, is_home=True) == screen_to_global(sx, sy)


def test_screen_to_team_away_is_rotated():
    # Bottom-left of pitch in screen → top-right corner in away team's frame.
    tx, ty = screen_to_team(PITCH_L, PITCH_B, is_home=False)
    assert abs(tx - 120.0) < 1e-9
    assert abs(ty - 80.0) < 1e-9


def test_screen_team_round_trip():
    for is_home in (True, False):
        for sx, sy in [(PITCH_L, PITCH_B), (PITCH_R, PITCH_T), (600, 400)]:
            tx, ty = screen_to_team(sx, sy, is_home=is_home)
            sx2, sy2 = team_to_screen(tx, ty, is_home=is_home)
            assert abs(sx2 - sx) < 1e-9
            assert abs(sy2 - sy) < 1e-9


# ── Delta: Screen ↔ Global ────────────────────────────────────────────────────


def test_screen_delta_to_global_full_pitch():
    """Full pitch in screen pixels → full pitch in standard units."""
    dx, dy = screen_delta_to_global(PITCH_R - PITCH_L, PITCH_T - PITCH_B)
    assert abs(dx - 120.0) < 1e-9
    assert abs(dy - 80.0) < 1e-9


def test_screen_delta_global_round_trip():
    for dx, dy in [(100.0, 50.0), (-200.0, 30.0)]:
        gdx, gdy = screen_delta_to_global(dx, dy)
        dx2, dy2 = global_delta_to_screen(gdx, gdy)
        assert abs(dx2 - dx) < 1e-9
        assert abs(dy2 - dy) < 1e-9


# ── Delta: Global ↔ Team ─────────────────────────────────────────────────────


def test_global_delta_to_team_home_is_identity():
    assert global_delta_to_team(3.0, -1.5, is_home=True) == (3.0, -1.5)


def test_global_delta_to_team_away_negates():
    assert global_delta_to_team(3.0, -1.5, is_home=False) == (-3.0, 1.5)


def test_global_team_delta_round_trip():
    for is_home in (True, False):
        for dx, dy in [(1.0, 0.5), (-2.0, 3.0)]:
            tdx, tdy = global_delta_to_team(dx, dy, is_home=is_home)
            dx2, dy2 = team_delta_to_global(tdx, tdy, is_home=is_home)
            assert abs(dx2 - dx) < 1e-9
            assert abs(dy2 - dy) < 1e-9


# ── Delta: Screen ↔ Team ─────────────────────────────────────────────────────


def test_screen_delta_to_team_home_same_as_global():
    dx, dy = 500.0, 325.0
    assert screen_delta_to_team(dx, dy, is_home=True) == screen_delta_to_global(dx, dy)


def test_screen_delta_to_team_away_negates():
    dx, dy = screen_delta_to_team(500.0, 325.0, is_home=False)
    gdx, gdy = screen_delta_to_global(500.0, 325.0)
    assert abs(dx - (-gdx)) < 1e-9
    assert abs(dy - (-gdy)) < 1e-9


def test_screen_team_delta_round_trip():
    for is_home in (True, False):
        for dx, dy in [(100.0, 50.0), (-200.0, 30.0)]:
            tdx, tdy = screen_delta_to_team(dx, dy, is_home=is_home)
            dx2, dy2 = team_delta_to_screen(tdx, tdy, is_home=is_home)
            assert abs(dx2 - dx) < 1e-9
            assert abs(dy2 - dy) < 1e-9
