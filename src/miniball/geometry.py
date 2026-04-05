"""Pure vector and pitch geometry utilities.

These functions operate on standard-pitch-space coordinates (X ∈ [0, 120],
Y ∈ [0, 80]) and have no side effects.  They are available to all layers
(simulation, AI, analytics, UI).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from miniball.config import (
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two ``(x, y)`` points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def norm(dx: float, dy: float) -> tuple[float, float]:
    """Normalise ``(dx, dy)`` to unit length.

    Returns ``(0.0, 0.0)`` when the vector magnitude is negligibly small to
    avoid division-by-zero errors.
    """
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)


def relative_position(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Return the position of ``b`` relative to ``a``."""
    return (b[0] - a[0], b[1] - a[1])


# Goal centres in team-relative (normalised) pitch coordinates.
# The own goal is always on the left (low X); the attacking goal on the right.
TEAM_GOAL_CENTER: tuple[float, float] = (0.0, STANDARD_PITCH_HEIGHT / 2)
TEAM_GOAL_LEFT_POST: tuple[float, float] = (
    0.0,
    STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2,
)
TEAM_GOAL_RIGHT_POST: tuple[float, float] = (
    0.0,
    STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2,
)
OPPOSITION_GOAL_CENTER: tuple[float, float] = (
    STANDARD_PITCH_WIDTH,
    STANDARD_PITCH_HEIGHT / 2,
)
OPPOSITION_GOAL_LEFT_POST: tuple[float, float] = (
    STANDARD_PITCH_WIDTH,
    STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2,
)
OPPOSITION_GOAL_RIGHT_POST: tuple[float, float] = (
    STANDARD_PITCH_WIDTH,
    STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2,
)
