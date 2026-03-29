"""AI interface contract for Miniball — state classes and the abstract base class.

This module defines the boundary between the game engine and any AI
implementation.  It intentionally contains *no* rendering or input logic:
only the data structures (state / action schemas) and the abstract
``BaseAI`` class that every AI must subclass.

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

State API
─────────
    # ── Player ──────────────────────────────────────────────────────────────
    player.number           → int
    player.is_teammate      → bool
    player.has_ball         → bool
    player.cooldown_timer   → float   # 0 = can receive ball
    player.location         → tuple[float, float]

    player.dist_to(target)           # distance to another player or (x, y)
    player.direction_to(target)      # unnormalised vector toward target
    player.closest_in(players)       # nearest player from a list

    # ── Ball ────────────────────────────────────────────────────────────────
    ball.location   → tuple[float, float]
    ball.velocity   → tuple[float, float]

    ball.projected_position(t)           # where will the ball be in t seconds?
    ball.position_when_crossing_x(x)     # where will it be when it crosses x?
    ball.closest_player_in(players)      # nearest player to the ball

    # ── Match ───────────────────────────────────────────────────────────────
    match.team_current_score        → int
    match.opposition_current_score  → int
    match.match_time_seconds        → float

    # ── Game (top-level state passed to get_actions) ─────────────────────────
    state.team          → list[PlayerState]   (this AI's players)
    state.opposition    → list[PlayerState]
    state.ball          → BallState
    state.match_state   → MatchState

    state.all_players   → list[PlayerState]   (team + opposition)
    state.ball_carrier  → PlayerState | None
    state.team_has_ball → bool

    state.players(
        teammates=True,           # include this AI's players
        opposition=True,          # include opposition players
        player_on_ball=True,      # include the player currently in possession
        players_on_cooldown=True, # include players with a non-zero cooldown timer
    ) → list[PlayerState]

Action schema
─────────────
    PlayerAction = {
        "direction": (dx, dy),  # desired displacement in standard pitch coords;
                                # clamped to PLAYER_SPEED * dt per frame — smaller
                                # vectors move the player by exactly that amount
        "strike":    bool,      # request to strike the ball; ignored if player has no ball
    }

    TeamActions = {
        player_number: PlayerAction,  # one entry per player; omitted players stand still
        ...
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypedDict

from miniball.config import BALL_DRAG

# ── State classes ─────────────────────────────────────────────────────────────


class PlayerState:
    """Snapshot of a single player's state in standard pitch coordinates."""

    def __init__(
        self,
        *,
        number: int,
        is_teammate: bool,
        has_ball: bool,
        cooldown_timer: float,
        location: tuple[float, float],
    ) -> None:
        self.number = number
        self.is_teammate = is_teammate
        self.has_ball = has_ball
        self.cooldown_timer = cooldown_timer
        self.location = location

    # ── Geometry helpers ──────────────────────────────────────────────────

    def dist_to(self, target: PlayerState | Sequence[float]) -> float:
        """Euclidean distance from this player to ``target``.

        ``target`` may be another ``PlayerState`` or any ``(x, y)``
        sequence (e.g. the return value of ``goal_center()``).
        """
        point = target.location if isinstance(target, PlayerState) else target
        return math.hypot(point[0] - self.location[0], point[1] - self.location[1])

    def direction_to(
        self, target: PlayerState | Sequence[float]
    ) -> tuple[float, float]:
        """Displacement vector from this player toward ``target``.

        The magnitude equals the distance to ``target``.  Pass the result
        directly as a ``PlayerAction`` direction: the engine will move the
        player at full speed when the distance exceeds ``PLAYER_SPEED * dt``,
        or by exactly this displacement when the player is already close.
        """
        point = target.location if isinstance(target, PlayerState) else target
        return (point[0] - self.location[0], point[1] - self.location[1])

    def closest_in(
        self,
        players: list[PlayerState],
        ignore_self: bool = True,
    ) -> PlayerState:
        """Return the player in ``players`` closest to this player.

        If ``ignore_self`` is ``True`` (default), the entry matching this
        player by both team and number is excluded.  No other filtering is
        applied — pass a mixed list to find the nearest regardless of team.
        """
        candidates = (
            [
                p
                for p in players
                if not (p.number == self.number and p.is_teammate == self.is_teammate)
            ]
            if ignore_self
            else list(players)
        )
        return min(candidates, key=lambda p: self.dist_to(p))

    def __repr__(self) -> str:
        team = "team" if self.is_teammate else "opp"
        return f"PlayerState({team}#{self.number} @ {self.location})"


class BallState:
    """Snapshot of the ball's state in standard pitch coordinates."""

    def __init__(
        self,
        *,
        location: tuple[float, float],
        velocity: tuple[float, float],
    ) -> None:
        self.location = location
        self.velocity = velocity

    # ── Physics helpers ───────────────────────────────────────────────────

    def projected_position(self, t: float) -> tuple[float, float]:
        """Project the ball's position after ``t`` seconds.

        Uses a continuous-drag approximation of the game engine's per-frame
        linear drag model:

            v(t) ≈ v₀ · exp(−BALL_DRAG · t)
            x(t) = x₀ + v₀/BALL_DRAG · (1 − exp(−BALL_DRAG · t))

        Ignores pitch boundaries and possession changes — accurate for
        short look-aheads (< 1 s) on an open pitch.  Negative ``t``
        returns the current position unchanged.
        """
        if t <= 0.0 or BALL_DRAG <= 0.0:
            return self.location
        decay = math.exp(-BALL_DRAG * t)
        factor = (1.0 - decay) / BALL_DRAG
        return (
            self.location[0] + self.velocity[0] * factor,
            self.location[1] + self.velocity[1] * factor,
        )

    def position_when_crossing_x(self, x: float) -> tuple[float, float] | None:
        """Project the ball's position when it first crosses ``x``.

        Returns ``None`` if the ball is stationary in x or already at ``x``.
        """
        if abs(self.velocity[0]) < 1e-6 or abs(x - self.location[0]) < 1e-6:
            return None
        t = (x - self.location[0]) / self.velocity[0]
        return self.projected_position(t)

    def closest_player_in(self, players: list[PlayerState]) -> PlayerState:
        """Return the player in ``players`` closest to the ball."""
        return min(players, key=lambda p: p.dist_to(self.location))

    def __repr__(self) -> str:
        return f"BallState(@ {self.location}, v={self.velocity})"


class MatchState:
    """Current match score and elapsed time."""

    def __init__(
        self,
        *,
        team_current_score: int,
        opposition_current_score: int,
        match_time_seconds: float,
    ) -> None:
        self.team_current_score = team_current_score
        self.opposition_current_score = opposition_current_score
        self.match_time_seconds = match_time_seconds

    def __repr__(self) -> str:
        return (
            f"MatchState({self.team_current_score}:{self.opposition_current_score}"
            f" @ {self.match_time_seconds:.1f}s)"
        )


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
        return next((p for p in self.all_players if p.has_ball), None)

    @property
    def team_has_ball(self) -> bool:
        """``True`` if any teammate currently holds the ball."""
        return any(p.has_ball for p in self.team)

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
            pool = [p for p in pool if not p.has_ball]
        if not players_on_cooldown:
            pool = [p for p in pool if p.cooldown_timer == 0.0]
        return pool


# ── Action TypedDicts (returned by AI, consumed by the game engine) ───────────


class PlayerAction(TypedDict):
    direction: tuple[
        float, float
    ]  # (dx, dy) in standard pitch coords; magnitude = speed fraction (0–1)
    strike: bool  # request to strike the ball; only meaningful for the player who has the ball


TeamActions = dict[
    int, PlayerAction
]  # player number → action; omitted players stand still


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
