"""Pitch-space coordinate transforms (Global ↔ Team).

Two normalised coordinate systems are used throughout the codebase:

Global (G)
    Team-agnostic normalised pitch space:
        x ∈ [0, STANDARD_PITCH_WIDTH],  y ∈ [0, STANDARD_PITCH_HEIGHT].
    No side-flip: the home team always attacks left → right
    (goal at x = STANDARD_PITCH_WIDTH) and the away team attacks right → left
    (goal at x = 0).

Team (T)
    Team-specific normalised pitch space in which the team in question always
    attacks left → right (goal at x = STANDARD_PITCH_WIDTH).  For the home
    team T ≡ G.  For the away team, T is the 180° rotation of G:
    (x, y) ↦ (W − x, H − y).

Screen-space transforms are in ``miniball.ui.coords``.
"""

from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH

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
