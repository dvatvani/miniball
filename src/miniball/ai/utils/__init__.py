"""Pure utility functions for AI implementations.

These functions operate on standard-pitch-space coordinates (X ∈ [0, 120],
Y ∈ [0, 80], team always attacks right) and have no side effects.  They are
free-standing so that AI authors can import and test them independently of
any ``BaseAI`` subclass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from miniball.ai.interface import BallState, PlayerState
from miniball.config import (
    BALL_DRAG,
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


def goal_center() -> tuple[float, float]:
    """Return the centre of the attacking goal in standard pitch coordinates.

    In the normalised view the attacking goal is always on the right side
    (high X), so this returns ``(STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT / 2)``.
    """
    return STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT / 2


def player_closest_to_point(
    players: list[PlayerState], point: Sequence[float]
) -> PlayerState:
    """Return the player in ``players`` closest to the point."""
    return min(players, key=lambda p: dist(p["location"], point))


def player_closest_to_ball(players: list[PlayerState], ball: BallState) -> PlayerState:
    """Return the player in ``players`` closest to the ball."""
    return player_closest_to_point(players, ball["location"])


def player_closest_to_player(
    player: PlayerState, players: list[PlayerState], ignore_self: bool = True
) -> PlayerState:
    """Return the same-team player in ``players`` closest to ``player``.

    Only players whose ``is_teammate`` flag matches ``player``'s are
    considered.  If ``ignore_self`` is ``True`` (the default), ``player``
    themselves is also excluded.
    """
    candidates = [p for p in players if p["is_teammate"] == player["is_teammate"]]
    if ignore_self:
        candidates = [p for p in candidates if p["number"] != player["number"]]
    return player_closest_to_point(candidates, player["location"])


def projected_ball_position(ball: BallState, t: float) -> tuple[float, float]:
    """Project the ball's position after ``t`` seconds.

    Uses a continuous-drag approximation of the discrete linear drag model
    applied each frame by the game engine:

        v(t) ≈ v₀ · exp(−BALL_DRAG · t)
        x(t) = x₀ + v₀ / BALL_DRAG · (1 − exp(−BALL_DRAG · t))

    This ignores pitch boundaries (wall bounces) and possession changes, so
    the prediction degrades for long time horizons or when the ball is near a
    wall.  For short look-aheads (< 1 s) on an open pitch the approximation
    is accurate to within a fraction of a pitch unit.

    Parameters
    ----------
    ball:
        Current ball state with ``location`` and ``velocity`` in standard
        pitch coordinates.
    t:
        Look-ahead time in seconds.  Negative values return the current
        position unchanged.

    Returns
    -------
    tuple[float, float]
        Predicted ``(x, y)`` position in standard pitch coordinates.
    """
    if t <= 0.0 or BALL_DRAG <= 0.0:
        return ball["location"][0], ball["location"][1]

    decay = math.exp(-BALL_DRAG * t)
    factor = (1.0 - decay) / BALL_DRAG

    x = ball["location"][0] + ball["velocity"][0] * factor
    y = ball["location"][1] + ball["velocity"][1] * factor
    return x, y


def projected_ball_position_when_crossing_x(
    ball: BallState, x: float
) -> tuple[float, float] | None:
    """Project the ball's position when it crosses the x value.

    Returns None if the ball is not moving.
    """
    if abs(ball["velocity"][0]) < 1e-6 or abs(x - ball["location"][0]) < 1e-6:
        return None
    t = (x - ball["location"][0]) / ball["velocity"][0]
    x_projected, y_projected = projected_ball_position(ball, t)
    return (x_projected, y_projected)
