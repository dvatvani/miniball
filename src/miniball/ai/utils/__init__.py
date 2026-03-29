"""Pure utility functions for AI implementations.

These functions operate on standard-pitch-space coordinates (X ∈ [0, 120],
Y ∈ [0, 80], team always attacks right) and have no side effects.  They are
free-standing so that AI authors can import and test them independently of
any ``BaseAI`` subclass.

Most player- and ball-centric operations are also available as methods on
``PlayerState`` and ``BallState`` directly (e.g. ``player.dist_to(target)``,
``ball.projected_position(t)``).  The free-standing forms here are kept for
use in list comprehensions and as arguments to ``min``/``max``/``sorted``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from miniball.ai.interface import BallState, PlayerState
from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH


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
    """Return the player in ``players`` closest to ``point``."""
    return min(players, key=lambda p: p.dist_to(point))


def player_closest_to_ball(players: list[PlayerState], ball: BallState) -> PlayerState:
    """Return the player in ``players`` closest to the ball."""
    return ball.closest_player_in(players)


def player_closest_to_player(
    player: PlayerState, players: list[PlayerState], ignore_self: bool = True
) -> PlayerState:
    """Return the player in ``players`` closest to ``player``.

    If ``ignore_self`` is ``True`` (the default), the entry matching ``player``
    by both team and number is excluded.  No other filtering is applied.
    """
    return player.closest_in(players, ignore_self=ignore_self)


def projected_ball_position(ball: BallState, t: float) -> tuple[float, float]:
    """Project the ball's position after ``t`` seconds.

    Delegates to ``ball.projected_position(t)`` — see ``BallState`` for full
    documentation of the drag model used.
    """
    return ball.projected_position(t)


def projected_ball_position_when_crossing_x(
    ball: BallState, x: float
) -> tuple[float, float] | None:
    """Project the ball's position when it first crosses ``x``.

    Returns ``None`` if the ball is stationary in x or already at ``x``.
    Delegates to ``ball.position_when_crossing_x(x)``.
    """
    return ball.position_when_crossing_x(x)
