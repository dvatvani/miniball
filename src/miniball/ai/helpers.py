"""AI interface and built-in implementations for Miniball.

Anatomy of an AI engine
────────────────────────
Subclass ``BaseAI`` and implement ``get_actions(state) -> TeamActions``.

An AI instance is a stateless-ish strategy object: it knows nothing about
which team it controls, which side of the pitch it occupies, or screen
pixels.  All of that is the game engine's responsibility.  The engine
prepares a normalised ``GameState`` and interprets the returned
``TeamActions``, handling coordinate conversion and team assignment itself.

Coordinate conventions
───────────────────────
All values (player locations, ball location, ball velocity, pitch bounds)
are in **standard pitch space**:

• X ∈ [0, 120], Y ∈ [0, 80]  (matches a standard 120 × 80 pitch).
• The team's own goal is always at the **left** (low X); the attacking
  goal is always at the **right** (high X).
• Y increases bottom → top; the attack's "left flank" is always the
  lower half of the pitch (low Y) regardless of physical side.
• Ball velocity is in the same normalised units per second.
• The ``move`` vector returned in ``PlayerAction`` is also in this frame.

The game engine applies a 180 ° rotation for the team that attacks left in
screen space, so AI implementations never need to think about it.

State schema
────────────
    GameState = {
        "players": [
            {
                "number":      int,
                "is_teammate": bool,
                "has_ball":    bool,
                "cooldown_timer": float, # seconds remaining on cooldown (0 = can receive ball)
                "location":    [x, y], # standard pitch coords
            },
            ...
        ],
        "ball": {
            "location":     [x, y],   # standard pitch coords
            "velocity":     [vx, vy], # standard pitch units / s
        },
        "match_state": {
            "team_current_score":        int,
            "opposition_current_score":  int,
            "match_time_seconds":        float,  # elapsed since kick-off
        }
    }

Action schema
─────────────
    PlayerAction = {
        "move":  [dx, dy],  # desired direction in standard pitch coords;
                            # magnitude used as speed fraction (0–1),
                            # clipped to 1 if larger
        "shoot": bool,      # request to shoot; ignored if player has no ball
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TypedDict

from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH

# ── State TypedDicts ──────────────────────────────────────────────────────────


class PlayerState(TypedDict):
    number: int
    is_teammate: bool
    has_ball: bool
    cooldown_timer: float  # seconds remaining on cooldown; 0 = can receive ball
    location: list[float]  # standard pitch coords [x, y]


class BallState(TypedDict):
    location: list[float]  # standard pitch coords [x, y]
    velocity: list[float]  # standard pitch units / s [vx, vy]


class MatchState(TypedDict):
    team_current_score: int
    opposition_current_score: int
    match_time_seconds: float  # elapsed since kick-off


class GameState(TypedDict):
    team: list[PlayerState]
    opposition: list[PlayerState]
    ball: BallState
    match_state: MatchState


class TeamActions(TypedDict):
    directions: dict[
        int, list[float]
    ]  # map of player numbers to direction vectors [dx, dy] in standard pitch coords; magnitude = speed fraction
    shoot: bool


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseAI(ABC):
    """Abstract base class for all AI engines.

    Instances carry no team information – the game engine assigns them to a
    team and handles all coordinate transforms.  The same instance can be
    used for both teams simultaneously if desired.
    """

    def __init__(self, formation: dict[int, list[float]]) -> None:
        self.formation = formation

    @abstractmethod
    def get_actions(self, state: GameState) -> TeamActions:
        """Return actions for every teammate in ``state``.

        The state is normalised so teammates always attack right.  Returned
        move vectors are interpreted in the same normalised frame.

        Parameters
        ----------
        state:
            Normalised game state.  ``is_teammate`` identifies this AI's
            players.

        Returns
        -------
        TeamActions
            Mapping of ``number -> direction`` and ``shoot``.  Omitted numbers
            default to ``[0, 0]`` for direction and ``False`` for shoot.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _goal_center() -> tuple[float, float]:
        """Centre of the attacking goal (always the right goal in normalised view)."""
        return STANDARD_PITCH_WIDTH, (STANDARD_PITCH_HEIGHT) / 2

    @staticmethod
    def _dist(a: list[float], b: list[float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def _norm(dx: float, dy: float) -> tuple[float, float]:
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)


def nearest_player(player: PlayerState, players: list[PlayerState]) -> PlayerState:
    """Return the nearest player to a given player from a list of players."""
    return min(players, key=lambda p: BaseAI._dist(p["location"], player["location"]))
