"""Screen-space coordinate transforms (Screen ↔ Global, Screen ↔ Team).

Screen (S)
    Raw arcade/pyglet pixel coordinates produced by the game engine.
    Origin at bottom-left corner of the window.
    Full window: x ∈ [0, SCREEN_W], y ∈ [0, SCREEN_H].
    Only the pitch area [PITCH_L, PITCH_R] × [PITCH_B, PITCH_T] is meaningful
    for player and ball positions.

For pitch-space transforms (Global ↔ Team) see ``miniball.coords``.
"""

from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH
from miniball.coords import global_to_team, team_to_global
from miniball.ui.config import PITCH_B, PITCH_L, PITCH_R, PITCH_T

_PITCH_PX_W: float = PITCH_R - PITCH_L
_PITCH_PX_H: float = PITCH_T - PITCH_B


# ── Position: Screen ↔ Global ─────────────────────────────────────────────────


def screen_to_global(sx: float, sy: float) -> tuple[float, float]:
    """Map a screen-space position to global pitch coordinates."""
    return (
        (sx - PITCH_L) / _PITCH_PX_W * STANDARD_PITCH_WIDTH,
        (sy - PITCH_B) / _PITCH_PX_H * STANDARD_PITCH_HEIGHT,
    )


def global_to_screen(gx: float, gy: float) -> tuple[float, float]:
    """Map global pitch coordinates to a screen-space position."""
    return (
        gx * _PITCH_PX_W / STANDARD_PITCH_WIDTH + PITCH_L,
        gy * _PITCH_PX_H / STANDARD_PITCH_HEIGHT + PITCH_B,
    )


# ── Position: Screen ↔ Team ───────────────────────────────────────────────────


def screen_to_team(sx: float, sy: float, is_home: bool) -> tuple[float, float]:
    """Map a screen-space position directly to the team-specific frame."""
    return global_to_team(*screen_to_global(sx, sy), is_home=is_home)


def team_to_screen(tx: float, ty: float, is_home: bool) -> tuple[float, float]:
    """Map a team-frame position directly to screen space."""
    return global_to_screen(*team_to_global(tx, ty, is_home=is_home))


# ── Delta: Screen ↔ Global ────────────────────────────────────────────────────


def screen_delta_to_global(dx: float, dy: float) -> tuple[float, float]:
    """Scale a screen-space vector to global pitch units (no sign change)."""
    return (
        dx * STANDARD_PITCH_WIDTH / _PITCH_PX_W,
        dy * STANDARD_PITCH_HEIGHT / _PITCH_PX_H,
    )


def global_delta_to_screen(dx: float, dy: float) -> tuple[float, float]:
    """Scale a global-pitch-unit vector to screen pixels (no sign change)."""
    return (
        dx * _PITCH_PX_W / STANDARD_PITCH_WIDTH,
        dy * _PITCH_PX_H / STANDARD_PITCH_HEIGHT,
    )


# ── Delta: Screen ↔ Team ─────────────────────────────────────────────────────


def screen_delta_to_team(dx: float, dy: float, is_home: bool) -> tuple[float, float]:
    """Scale and optionally flip a screen-space vector into the team frame."""
    from miniball.coords import global_delta_to_team

    return global_delta_to_team(*screen_delta_to_global(dx, dy), is_home=is_home)


def team_delta_to_screen(dx: float, dy: float, is_home: bool) -> tuple[float, float]:
    """Convert a team-frame vector to screen pixels."""
    from miniball.coords import team_delta_to_global

    return global_delta_to_screen(*team_delta_to_global(dx, dy, is_home=is_home))
