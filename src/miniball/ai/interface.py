"""AI interface contract for Miniball — TypedDicts and the abstract base class.

This module defines the boundary between the game engine and any AI
implementation.  It intentionally contains *no* gameplay logic: only the
data structures (state / action schemas) and the abstract ``BaseAI`` class
that every AI must subclass.

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
        "direction": (dx, dy),  # desired direction in standard pitch coords;
                                # magnitude used as speed fraction (0–1),
                                # clipped to 1 if larger
        "strike":    bool,      # request to strike the ball; ignored if player has no ball
    }

    TeamActions = {
        "actions": {
            player_number: PlayerAction,  # one entry per player; omitted players stand still
            ...
        }
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

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


class PlayerAction(TypedDict):
    direction: tuple[
        float, float
    ]  # (dx, dy) in standard pitch coords; magnitude = speed fraction (0–1)
    strike: bool  # request to strike the ball; only meaningful for the player who has the ball


class TeamActions(TypedDict):
    actions: dict[
        int, PlayerAction
    ]  # player number → per-player action; omitted players stand still


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
            Per-player actions keyed by player number.  Omitted players stand
            still and do not strike.
        """
        ...
