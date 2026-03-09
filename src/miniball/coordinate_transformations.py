"""Coordinate system conversions for Miniball.

Three coordinate systems are used throughout the codebase:

Screen (S)
    Raw arcade/pyglet pixel coordinates produced by the game engine.
    Origin at bottom-left corner of the window.
    Full window: x ∈ [0, SCREEN_W], y ∈ [0, SCREEN_H].
    Only the pitch area [PITCH_L, PITCH_R] × [PITCH_B, PITCH_T] is meaningful
    for player and ball positions.

Global (G)
    Team-agnostic normalised pitch space, obtained by a linear mapping of the
    pitch region in screen space onto standard pitch dimensions:
        x ∈ [0, STANDARD_PITCH_WIDTH],  y ∈ [0, STANDARD_PITCH_HEIGHT].
    No side-flip is applied, so the home team always attacks left → right
    (goal at x = STANDARD_PITCH_WIDTH) and the away team attacks right → left
    (goal at x = 0).  This is the shared reference frame used for data storage
    and cross-team visualisation.

Team (T)
    Team-specific normalised pitch space in which the team in question always
    attacks left → right (goal at x = STANDARD_PITCH_WIDTH), regardless of
    physical side.  For the home team T ≡ G.  For the away team, T is the 180°
    rotation of G:  (x, y) ↦ (W − x, H − y).
    This is the frame presented to AI engines via ``_build_game_state``.

Position conversion functions (6)
    screen_to_global   / global_to_screen       no team context needed
    global_to_team     / team_to_global          require ``is_home: bool``
    screen_to_team     / team_to_screen          require ``is_home: bool``

Delta (direction or velocity vector) conversion functions (6)
    Deltas apply scale changes and, where relevant, sign flips, but *no*
    positional offsets, making them correct for velocity vectors and
    normalised direction inputs as well as positional differences.

    screen_delta_to_global   / global_delta_to_screen   no team context needed
    global_delta_to_team     / team_delta_to_global      require ``is_home: bool``
    screen_delta_to_team     / team_delta_to_screen      require ``is_home: bool``
"""

from miniball.config import (
    PITCH_B,
    PITCH_L,
    PITCH_R,
    PITCH_T,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)

# Pitch extent in screen pixels – used as scale factors throughout.
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


# ── Position: Global ↔ Team ───────────────────────────────────────────────────


def global_to_team(gx: float, gy: float, is_home: bool) -> tuple[float, float]:
    """Convert a global-frame position to the team-specific frame.

    For the home team the two frames are identical.  For the away team the
    180° pitch rotation (x ↦ W − x, y ↦ H − y) is applied.
    """
    if is_home:
        return (gx, gy)
    return (STANDARD_PITCH_WIDTH - gx, STANDARD_PITCH_HEIGHT - gy)


def team_to_global(tx: float, ty: float, is_home: bool) -> tuple[float, float]:
    """Convert a team-frame position to the global frame.

    Inverse of ``global_to_team``; the 180° rotation is its own inverse.
    """
    if is_home:
        return (tx, ty)
    return (STANDARD_PITCH_WIDTH - tx, STANDARD_PITCH_HEIGHT - ty)


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


# ── Delta: Global ↔ Team ─────────────────────────────────────────────────────


def global_delta_to_team(dx: float, dy: float, is_home: bool) -> tuple[float, float]:
    """Apply the team-frame sign convention to a global direction/velocity vector.

    For the home team the two frames are identical.  For the away team both
    components are negated (consequence of the 180° rotation).
    """
    if is_home:
        return (dx, dy)
    return (-dx, -dy)


def team_delta_to_global(dx: float, dy: float, is_home: bool) -> tuple[float, float]:
    """Convert a team-frame direction/velocity vector to the global frame.

    Inverse of ``global_delta_to_team``; negation is its own inverse.
    """
    if is_home:
        return (dx, dy)
    return (-dx, -dy)


# ── Delta: Screen ↔ Team ─────────────────────────────────────────────────────


def screen_delta_to_team(dx: float, dy: float, is_home: bool) -> tuple[float, float]:
    """Scale and optionally flip a screen-space vector into the team frame."""
    return global_delta_to_team(*screen_delta_to_global(dx, dy), is_home=is_home)


def team_delta_to_screen(dx: float, dy: float, is_home: bool) -> tuple[float, float]:
    """Convert a team-frame vector to screen pixels."""
    return global_delta_to_screen(*team_delta_to_global(dx, dy, is_home=is_home))
