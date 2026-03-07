"""AI interface and built-in implementations for Miniball.

Anatomy of an AI engine
────────────────────────
Subclass ``BaseAI`` and implement ``get_actions(state) -> TeamActions``.

The game calls ``get_actions`` once per frame for each registered AI,
passing a ``GameState`` dict in which ``is_teammate`` is set from that
AI's perspective.  The method should return a ``TeamActions`` mapping
(``player_id -> PlayerAction``) for every player on its team.  Missing
player IDs are treated as "stand still, don't shoot".

State schema
────────────
    GameState = {
        "players": [
            {
                "player_id":   str,           # e.g. "A1", "B3"
                "team":        "A" | "B",
                "is_teammate": bool,
                "has_ball":    bool,
                "on_cooldown": bool,          # True = cannot gain ball possession
                "location":    [x, y],        # screen pixels
                "facing":      float,         # radians
            },
            ...
        ],
        "ball": {
            "location":     [x, y],
            "velocity":     [vx, vy],
            "possessed_by": str | None,       # player_id, or None if free
        },
        "pitch": {
            "left":                float,
            "right":               float,
            "bottom":              float,
            "top":                 float,
            "goal_height":         float,
            "attacking_direction": 1 | -1,    # +1 = attack right, -1 = left
        },
    }

Action schema
─────────────
    PlayerAction = {
        "move":  [dx, dy],   # desired direction; magnitude is used as speed
                             # fraction (0–1); will be normalised if > 1
        "shoot": bool,       # request to shoot; ignored if player has no ball
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TypedDict


# ── State TypedDicts ──────────────────────────────────────────────────────────


class PlayerState(TypedDict):
    player_id: str
    team: str
    is_teammate: bool
    has_ball: bool
    on_cooldown: bool  # True = cannot gain ball possession (can still move)
    location: list[float]
    facing: float


class BallState(TypedDict):
    location: list[float]
    velocity: list[float]
    possessed_by: str | None


class PitchInfo(TypedDict):
    left: float
    right: float
    bottom: float
    top: float
    goal_height: float
    attacking_direction: int  # +1 = attack right, -1 = attack left


class GameState(TypedDict):
    players: list[PlayerState]
    ball: BallState
    pitch: PitchInfo


class PlayerAction(TypedDict):
    move: list[float]  # [dx, dy] – normalised in game if magnitude > 1
    shoot: bool


# player_id → PlayerAction
TeamActions = dict[str, PlayerAction]


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseAI(ABC):
    """Abstract base class for all AI engines.

    Parameters
    ----------
    team:
        ``"A"`` or ``"B"`` – which team this engine controls.
    """

    def __init__(self, team: str) -> None:
        self.team = team

    @abstractmethod
    def get_actions(self, state: GameState) -> TeamActions:
        """Return actions for every player on this team.

        Parameters
        ----------
        state:
            Full game state with ``is_teammate`` set from this team's
            perspective.

        Returns
        -------
        TeamActions
            Mapping of ``player_id -> PlayerAction``.  Omitted player IDs
            default to ``{"move": [0, 0], "shoot": False}``.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _goal_center(self, pitch: PitchInfo) -> tuple[float, float]:
        """(x, y) of the centre of the goal this team is attacking."""
        x = pitch["right"] if pitch["attacking_direction"] == 1 else pitch["left"]
        y = (pitch["top"] + pitch["bottom"]) / 2
        return x, y

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
            p["player_id"]: {"move": [0.0, 0.0], "shoot": False}
            for p in state["players"]
            if p["is_teammate"]
        }


class BaselineAI(BaseAI):
    """Simple rule-based AI.

    Decision hierarchy (evaluated per player, per frame):

    1. **Has the ball** → dribble straight toward the attacking goal;
       shoot when within ``SHOOT_RANGE`` pixels of the goal centre.
    2. **No ball / opposition has it** → press toward the ball at full speed.
    3. **Teammate has the ball** → drift back toward home position to avoid
       crowding the ball carrier and leave space open.

    Home positions are cached from each player's location on the first frame
    so the AI naturally inherits whatever starting layout the game uses.
    """

    SHOOT_RANGE: float = 280.0  # px to goal centre at which the AI shoots
    HOME_DEADBAND: float = 20.0  # px – don't move if already close to home

    def __init__(self, team: str) -> None:
        super().__init__(team)
        # Populated on the first get_actions call from each player's initial position
        self._home: dict[str, list[float]] = {}

    def get_actions(self, state: GameState) -> TeamActions:
        # Snapshot starting positions once so we can return to them
        if not self._home:
            for p in state["players"]:
                if p["is_teammate"]:
                    self._home[p["player_id"]] = list(p["location"])

        pitch = state["pitch"]
        gx, gy = self._goal_center(pitch)
        ball_loc = state["ball"]["location"]

        teammate_has_ball = any(
            p["is_teammate"] and p["has_ball"] for p in state["players"]
        )

        actions: TeamActions = {}

        for p in state["players"]:
            if not p["is_teammate"]:
                continue

            pid = p["player_id"]
            px, py = p["location"]

            if p["has_ball"]:
                # ── Dribble toward goal; shoot when close enough ───────────
                dx, dy = self._norm(gx - px, gy - py)
                dist_to_goal = self._dist([px, py], [gx, gy])
                actions[pid] = {
                    "move": [dx, dy],
                    "shoot": dist_to_goal < self.SHOOT_RANGE,
                }

            elif teammate_has_ball:
                # ── Drift back to home position to open up space ───────────
                home = self._home.get(pid, [px, py])
                if self._dist([px, py], home) > self.HOME_DEADBAND:
                    dx, dy = self._norm(home[0] - px, home[1] - py)
                    actions[pid] = {"move": [dx, dy], "shoot": False}
                else:
                    actions[pid] = {"move": [0.0, 0.0], "shoot": False}

            else:
                # ── Press toward the ball ──────────────────────────────────
                dx, dy = self._norm(ball_loc[0] - px, ball_loc[1] - py)
                actions[pid] = {"move": [dx, dy], "shoot": False}

        return actions
