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
                "on_cooldown": bool,   # True = cannot gain ball possession
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
            "seconds_left":              float,
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
    on_cooldown: bool  # True = cannot gain ball possession (can still move)
    location: list[float]  # standard pitch coords [x, y]


class BallState(TypedDict):
    location: list[float]  # standard pitch coords [x, y]
    velocity: list[float]  # standard pitch units / s [vx, vy]


class MatchState(TypedDict):
    team_current_score: int
    opposition_current_score: int
    seconds_left: float


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


# ── Built-in implementations ──────────────────────────────────────────────────


class StationaryAI(BaseAI):
    """Every player stands still and never shoots.

    Useful as a neutral placeholder while you develop the real AI.
    """

    def get_actions(self, state: GameState) -> TeamActions:
        return {
            "directions": {p["number"]: [0.0, 0.0] for p in state["team"]},
            "shoot": False,
        }


class BaselineAI(BaseAI):
    """Simple rule-based AI.

    Decision hierarchy (evaluated per player, per frame):

    1. **Has the ball** → dribble straight toward the attacking goal;
       shoot when within ``SHOOT_RANGE`` normalised units of the goal centre.
    2. **No ball / opposition has it** → press toward the ball at full speed.
    3. **Teammate has the ball** → drift back toward home position to avoid
       crowding the ball carrier and leave space open.

    Home positions are cached from each player's location on the first frame
    so the AI naturally inherits whatever starting layout the game uses.

    Because the state is always normalised to attack right and expressed in
    standard pitch coordinates, this class contains no team-side, pixel, or
    coordinate-direction logic.
    """

    SHOOT_RANGE: float = 28.0  # normalised units to goal centre at which the AI shoots
    HOME_DEADBAND: float = 2.0  # normalised units – don't move if already close to home

    def get_actions(self, state: GameState) -> TeamActions:
        gx, gy = self._goal_center()
        ball_loc = state["ball"]["location"]

        teammate_has_ball = any(
            p["is_teammate"] and p["has_ball"] for p in state["team"]
        )

        shoot = False
        directions = {}

        for p in state["team"]:
            pid = p["number"]
            px, py = p["location"]

            if p["has_ball"]:
                # ── Dribble toward goal; shoot when close enough ───────────
                dx, dy = self._norm(gx - px, gy - py)
                dist_to_goal = self._dist([px, py], [gx, gy])
                directions[pid] = [dx, dy]
                shoot = dist_to_goal < self.SHOOT_RANGE

            elif teammate_has_ball:
                # ── Drift back to home position to open up space ───────────
                formation_location = self.formation.get(pid, [px, py])
                if self._dist([px, py], formation_location) > self.HOME_DEADBAND:
                    dx, dy = self._norm(
                        formation_location[0] - px, formation_location[1] - py
                    )
                    directions[pid] = [dx, dy]
                else:
                    directions[pid] = [0.0, 0.0]

            else:
                # ── Press toward the ball ──────────────────────────────────
                dx, dy = self._norm(ball_loc[0] - px, ball_loc[1] - py)
                directions[pid] = [dx, dy]

        return {
            "directions": directions,
            "shoot": shoot,
        }
