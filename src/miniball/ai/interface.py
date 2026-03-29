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
    state.team          → list[PlayerState]   (this AI's players)
    state.opposition    → list[PlayerState]
    state.ball          → BallState
    state.match_state   → MatchState

    state.all_players   → list[PlayerState]   (team + opposition)
    state.ball_carrier  → PlayerState | None
    state.team_has_ball → bool

    state.players(
        teammates=True,        # include this AI's players
        opposition=True,       # include opposition players
        player_on_ball=True,   # include the player currently in possession
        players_on_cooldown=True,  # include players with a non-zero cooldown timer
    ) → list[PlayerState]

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
    location: tuple[float, float]  # standard pitch coords (x, y)


class BallState(TypedDict):
    location: tuple[float, float]  # standard pitch coords (x, y)
    velocity: tuple[float, float]  # standard pitch units / s (vx, vy)


class MatchState(TypedDict):
    team_current_score: int
    opposition_current_score: int
    match_time_seconds: float  # elapsed since kick-off


class GameState:
    """Rich game state passed to every ``get_actions`` call.

    Wraps the per-frame snapshot with convenience properties and a flexible
    player-filtering helper so AI implementations can express intent clearly
    without boilerplate filtering loops.

    Attributes
    ----------
    team:
        This AI's players, normalised so the team always attacks right.
    opposition:
        The opposing team's players in the same coordinate frame.
    ball:
        Current ball position and velocity.
    match_state:
        Scores and elapsed time.
    """

    def __init__(
        self,
        team: list[PlayerState],
        opposition: list[PlayerState],
        ball: BallState,
        match_state: MatchState,
    ) -> None:
        self.team = team
        self.opposition = opposition
        self.ball = ball
        self.match_state = match_state

    @property
    def all_players(self) -> list[PlayerState]:
        """All players on the pitch — teammates first, then opposition."""
        return self.team + self.opposition

    @property
    def ball_carrier(self) -> PlayerState | None:
        """The player currently in possession of the ball, or ``None``."""
        return next((p for p in self.all_players if p["has_ball"]), None)

    @property
    def team_has_ball(self) -> bool:
        """``True`` if any teammate currently holds the ball."""
        return any(p["has_ball"] for p in self.team)

    def players(
        self,
        *,
        teammates: bool = True,
        opposition: bool = True,
        player_on_ball: bool = True,
        players_on_cooldown: bool = True,
    ) -> list[PlayerState]:
        """Return a filtered list of players.

        Parameters
        ----------
        teammates:
            Include this AI's players.
        opposition:
            Include opposition players.
        player_on_ball:
            Include the player currently in possession of the ball.
        players_on_cooldown:
            Include players whose ``cooldown_timer`` is non-zero (i.e. they
            recently struck the ball and cannot yet receive a pass).
        """
        pool: list[PlayerState] = []
        if teammates:
            pool.extend(self.team)
        if opposition:
            pool.extend(self.opposition)
        if not player_on_ball:
            pool = [p for p in pool if not p["has_ball"]]
        if not players_on_cooldown:
            pool = [p for p in pool if p["cooldown_timer"] == 0.0]
        return pool


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

    def __init__(self, formation: dict[int, tuple[float, float]]) -> None:
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
