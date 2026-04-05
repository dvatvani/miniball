"""AI-specific utility functions.

General-purpose vector and pitch geometry helpers live in
``miniball.geometry``.  This module contains only helpers that depend on
AI-layer types (e.g. ``PlayerState``).
"""

from __future__ import annotations

from collections.abc import Sequence

from miniball.ai.interface import PlayerState


def player_closest_to_point(
    players: list[PlayerState], point: Sequence[float]
) -> PlayerState:
    """Return the player in ``players`` closest to ``point``."""
    return min(players, key=lambda p: p.dist_to(point))
