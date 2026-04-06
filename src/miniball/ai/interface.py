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
    player.is_home          → bool   # True for the home team, regardless of AI perspective
    player.has_ball         → bool
    player.cooldown_timer   → float   # 0 = can receive ball
    player.location         → tuple[float, float]   # team-relative coords (attacks right)
    player.global_location  → tuple[float, float]   # shared global frame (home attacks right)

    player.dist_to(target)           # distance to another player or (x, y)
    player.direction_to(target)      # unnormalised vector toward target
    player.closest_player(players)   # nearest player from a list (by distance)
    player.intercept_time(ball)      # seconds until player can reach the ball
    player.intercept_point(ball)     # position where player will meet the ball

    # ── Ball ────────────────────────────────────────────────────────────────
    ball.location   → tuple[float, float]
    ball.velocity   → tuple[float, float]

    ball.projected_position(t)                    # where will the ball be in t seconds?
    ball.position_when_crossing_x(x)             # where will it be when it crosses x?
    ball.trace_path()                             # list[BallPathPoint]: start + bounces + stop
    ball.closest_player(players)                  # nearest player to the ball (by distance)
    ball.interception_times(players)              # [(time, player), …] sorted fastest-first
    ball.fastest_interceptor(players)             # player who reaches ball soonest
    ball.intercept_advantage(player, opponents)   # seconds player beats nearest opponent

    # ── Match ───────────────────────────────────────────────────────────────
    match.team_current_score        → int
    match.opposition_current_score  → int
    match.match_time_seconds        → float

    # ── Game (top-level state passed to get_actions) ─────────────────────────
    state.team          → list[PlayerState]   (this AI's players)
    state.opposition    → list[PlayerState]
    state.ball          → BallState
    state.match_state   → MatchState
    state.is_home       → bool                (True if this AI controls the home team)

    state.all_players        → list[PlayerState]   (team + opposition)
    state.ball_carrier       → PlayerState | None
    state.team_has_ball      → bool
    state.global_ball_location   → tuple[float, float]   (ball in shared global frame)
    state.global_ball_velocity   → tuple[float, float]

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
from typing import NamedTuple, TypedDict

from miniball.config import (
    BALL_DRAG,
    BALL_RADIUS,
    PLAYER_SPEED,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)

_PLAYER_SPEED_SQ: float = PLAYER_SPEED * PLAYER_SPEED
# Ball speed (units/s) below which it is treated as effectively stopped
_BALL_STOP_SPEED: float = 0.1

# Effective pitch boundaries for the ball centre (keeps ball_radius from walls)
_WALL_LEFT: float = BALL_RADIUS
_WALL_RIGHT: float = STANDARD_PITCH_WIDTH - BALL_RADIUS
_WALL_BOTTOM: float = BALL_RADIUS
_WALL_TOP: float = STANDARD_PITCH_HEIGHT - BALL_RADIUS

# Goal opening: y range of the goal mouth (same for both ends of the pitch)
_GOAL_LO: float = STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2
_GOAL_HI: float = STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2


def _time_to_wall(pos0: float, vel0: float, wall: float) -> float | None:
    """Time for a 1-D coordinate to reach *wall* under linear drag.

    Assumes *vel0* is moving toward *wall* (same sign as ``wall - pos0``).
    Returns ``None`` when drag brings the ball to rest before it reaches the
    wall (i.e. the maximum displacement ``vel0 / BALL_DRAG`` falls short).
    """
    arg = 1.0 - (wall - pos0) * BALL_DRAG / vel0
    return -math.log(arg) / BALL_DRAG if arg > 0.0 else None


def _segment_intercept_t(
    px: float,
    py: float,
    seg_lx: float,
    seg_ly: float,
    seg_vx: float,
    seg_vy: float,
    seg_t0: float,
) -> float | None:
    """Smallest non-negative segment-local time at which a player can intercept the ball.

    Solves the generalised quadratic arising when the ball segment starts at
    cumulative time *seg_t0* rather than zero:

        |seg_loc + seg_vel·t_seg − player|² = PLAYER_SPEED² · (seg_t0 + t_seg)²

    Expanding and rearranging:

        A·t² + B·t + C = 0
        A = |V|²  − PLAYER_SPEED²
        B = 2·(d·V − PLAYER_SPEED²·seg_t0)   (d = seg_loc − player)
        C = |d|²  − PLAYER_SPEED²·seg_t0²

    When *seg_t0 = 0* this reduces to the original single-segment quadratic.

    Returns ``None`` when no non-negative solution exists.
    """
    dx = seg_lx - px
    dy = seg_ly - py

    A = seg_vx * seg_vx + seg_vy * seg_vy - _PLAYER_SPEED_SQ
    B = 2.0 * (dx * seg_vx + dy * seg_vy - _PLAYER_SPEED_SQ * seg_t0)
    C = dx * dx + dy * dy - _PLAYER_SPEED_SQ * seg_t0 * seg_t0

    # C ≤ 0 means the player can already reach the segment start by seg_t0.
    if C <= 1e-9:
        return 0.0

    if abs(A) < 1e-9:
        # Ball speed ≈ player speed → linear equation B·t + C = 0
        if abs(B) < 1e-9:
            return None
        t = -C / B
        return t if t >= 0.0 else None

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)
    positives = [t for t in (t1, t2) if t >= 0.0]
    return min(positives) if positives else None


# ── State classes ─────────────────────────────────────────────────────────────


class PlayerState:
    """Snapshot of a single player's state in standard pitch coordinates."""

    def __init__(
        self,
        *,
        number: int,
        is_teammate: bool,
        is_home: bool,
        has_ball: bool,
        cooldown_timer: float,
        location: tuple[float, float],
    ) -> None:
        self.number = number
        self.is_teammate = is_teammate
        self.is_home = is_home
        self.has_ball = has_ball
        self.cooldown_timer = cooldown_timer
        self.location = location

    @property
    def global_location(self) -> tuple[float, float]:
        """Player's position in the shared global frame (home team attacks right).

        ``location`` is stored in the *perspective* frame of the ``GameState``
        that created this player — i.e. the frame where this AI's team always
        attacks right.  For a home-team ``GameState`` that is already the global
        frame; for an away-team ``GameState`` coordinates are rotated 180°.

        The perspective can be recovered without storing it explicitly:
        ``state_is_home == (is_teammate == is_home)``, which holds for all four
        combinations of perspective × team membership.
        """
        state_is_home = self.is_teammate == self.is_home
        if state_is_home:
            return self.location
        return (
            STANDARD_PITCH_WIDTH - self.location[0],
            STANDARD_PITCH_HEIGHT - self.location[1],
        )

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

    def intercept_time(self, ball: BallState) -> float:
        """Minimum time (seconds) until this player can reach the ball.

        Uses a constant-velocity approximation within each segment of the
        ball's traced path (see :meth:`BallState.trace_path`).  Wall bounces
        are handled by iterating over segments in order and solving the
        generalised quadratic for each one:

            A·t² + B·t + C = 0
            A = |V|²  − PLAYER_SPEED²
            B = 2·(d·V − PLAYER_SPEED²·t₀)   (d = seg_loc − player, t₀ = seg start time)
            C = |d|²  − PLAYER_SPEED²·t₀²

        The first segment whose solution lies within the segment's duration
        determines the intercept.  If the ball is stopped before the player
        arrives, the time to walk to the stopped position is returned instead.
        """
        px, py = self.location
        path = ball.trace_path()

        for i in range(len(path) - 1):
            seg = path[i]
            seg_duration = path[i + 1].time - seg.time
            lx, ly = seg.location
            vx, vy = seg.velocity

            t_seg = _segment_intercept_t(px, py, lx, ly, vx, vy, seg.time)
            if t_seg is not None and t_seg <= seg_duration + 1e-9:
                return seg.time + t_seg

        # Ball permanently stopped at path[-1]; player walks there.
        stop = path[-1]
        slx, sly = stop.location
        walk_time = math.hypot(slx - px, sly - py) / PLAYER_SPEED
        return max(walk_time, stop.time)

    def intercept_point(self, ball: BallState) -> tuple[float, float]:
        """Position where this player will meet the ball at the earliest opportunity.

        Traces the ball's path (including wall bounces) and returns the point
        on the first reachable segment at which the player can intercept.
        Pass the result to ``direction_to`` to move toward it::

            action = player.direction_to(player.intercept_point(ball))
        """
        px, py = self.location
        path = ball.trace_path()

        for i in range(len(path) - 1):
            seg = path[i]
            seg_duration = path[i + 1].time - seg.time
            lx, ly = seg.location
            vx, vy = seg.velocity

            t_seg = _segment_intercept_t(px, py, lx, ly, vx, vy, seg.time)
            if t_seg is not None and t_seg <= seg_duration + 1e-9:
                return (lx + vx * t_seg, ly + vy * t_seg)

        return path[-1].location

    def closest_player(
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
                if not (p.number == self.number and p.is_home == self.is_home)
            ]
            if ignore_self
            else list(players)
        )
        return min(candidates, key=lambda p: self.dist_to(p))

    def angle_to(
        self,
        target: PlayerState | Sequence[float],
        subject: PlayerState | Sequence[float],
    ) -> float:
        """Signed angle (radians) at this player between the directions to ``target`` and ``subject``.

        Measures the angle target–player–subject, i.e. the angle at this
        player's position between the ray pointing toward ``target`` and the
        ray pointing toward ``subject``.  Positive values are
        counter-clockwise; the result lies in ``(-π, π]``.

        Both arguments may be a ``PlayerState`` or any ``(x, y)`` sequence
        (e.g. the return value of ``goal_center()``).

        Example — check whether ``subject`` is within 30° of the direction to
        ``target``::

            abs(player.angle_to(target, subject)) < math.radians(30)
        """
        tx, ty = target.location if isinstance(target, PlayerState) else target
        sx, sy = subject.location if isinstance(subject, PlayerState) else subject
        px, py = self.location
        atx, aty = tx - px, ty - py
        asx, asy = sx - px, sy - py
        return math.atan2(atx * asy - aty * asx, atx * asx + aty * asy)

    def __repr__(self) -> str:
        side = "home" if self.is_home else "away"
        role = "team" if self.is_teammate else "opp"
        return f"PlayerState({side}/{role}#{self.number} @ {self.location})"


class BallPathPoint(NamedTuple):
    """A single waypoint in the ball's traced path.

    *location* and *velocity* are in the same coordinate frame as the parent
    ``BallState``.  *time* is the cumulative number of seconds from the moment
    ``trace_path`` was called.

    For bounce waypoints the velocity is the **post-bounce** value; for the
    final (stopped) waypoint the velocity is ``(0.0, 0.0)``.
    """

    location: tuple[float, float]
    velocity: tuple[float, float]
    time: float  # cumulative seconds from trace start


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

    def trace_path(self, max_bounces: int = 10) -> list[BallPathPoint]:
        """Trace the ball's full path under drag, including wall bounces.

        Returns a list of :class:`BallPathPoint` waypoints::

            [start, bounce₁, bounce₂, …, stop]

        * **start** — the ball's current position and velocity (``time = 0``).
        * **bounce** — position and *post-bounce* velocity at each wall collision.
        * **stop** — where the ball effectively comes to rest (speed below
          ``_BALL_STOP_SPEED``); velocity is ``(0.0, 0.0)``.

        Each waypoint's ``time`` is the cumulative seconds from *now*.

        The list always contains at least two entries (start + stop), even for
        a stationary or very slow ball.

        Bounce reflection rules:
        * Left / right wall  → x-velocity is negated.
        * Top / bottom wall  → y-velocity is negated.
        * Goal openings      → no bounce; the trace stops and the final waypoint
          is placed at the ball's drag-model rest position (which will overshoot
          the goal line — this is intentional).

        *max_bounces* caps the number of wall reflections computed (default 10)
        as a safety limit; in practice a ball will decelerate to near-rest long
        before this.
        """
        x, y = self.location
        vx, vy = self.velocity
        t_cum = 0.0

        path: list[BallPathPoint] = [BallPathPoint((x, y), (vx, vy), t_cum)]

        for _ in range(max_bounces):
            speed = math.hypot(vx, vy)
            if speed < _BALL_STOP_SPEED:
                # Already effectively stopped; path ends at current point.
                break

            # Rest position under drag: x₀ + vx/k (as t → ∞)
            rest_x = x + vx / BALL_DRAG
            rest_y = y + vy / BALL_DRAG

            if (
                _WALL_LEFT <= rest_x <= _WALL_RIGHT
                and _WALL_BOTTOM <= rest_y <= _WALL_TOP
            ):
                # Ball decelerates to rest inside the pitch — append stop point.
                t_stop = math.log(speed / _BALL_STOP_SPEED) / BALL_DRAG
                decay = math.exp(-BALL_DRAG * t_stop)
                factor = (1.0 - decay) / BALL_DRAG
                path.append(
                    BallPathPoint(
                        (x + vx * factor, y + vy * factor),
                        (0.0, 0.0),
                        t_cum + t_stop,
                    )
                )
                return path

            # Find time to the relevant wall in each axis (only when moving toward it).
            t_x_raw = (
                _time_to_wall(x, vx, _WALL_RIGHT if vx > 0.0 else _WALL_LEFT)
                if abs(vx) > 1e-9
                else None
            )
            t_y = (
                _time_to_wall(y, vy, _WALL_TOP if vy > 0.0 else _WALL_BOTTOM)
                if abs(vy) > 1e-9
                else None
            )

            # Check whether the x-wall hit is actually a goal opening.
            # Compute the ball's y at the moment it reaches the left/right wall.
            # If that y falls in the goal mouth and the ball gets there before any
            # y-wall hit, stop tracing (no bounce off a goal); the fallback below
            # will append the drag-model rest point behind the goal line.
            # If a y-wall hit comes first, skip the x-wall this iteration so the
            # y-bounce is processed; the goal check repeats on the next pass.
            t_x = t_x_raw
            if t_x_raw is not None:
                decay_x = math.exp(-BALL_DRAG * t_x_raw)
                factor_x = (1.0 - decay_x) / BALL_DRAG
                y_at_x_wall = y + vy * factor_x
                if _GOAL_LO <= y_at_x_wall <= _GOAL_HI:
                    if t_y is None or t_x_raw <= t_y:
                        # Ball enters goal before any y-wall — stop tracing.
                        break
                    # y-wall hit comes first; defer the goal check to next pass.
                    t_x = None

            if t_x is None and t_y is None:
                # Neither wall reachable under drag (shouldn't happen when rest is outside).
                break

            # Take the earliest wall hit; prefer x-wall on tie.
            if t_x is not None and (t_y is None or t_x <= t_y):
                t_hit, hit_x_wall = t_x, True
            else:
                # t_y is not None here: guaranteed by the `break` above which
                # exits when both are None, so the else branch is only reached
                # when t_x is None (→ t_y must be non-None) or t_y < t_x.
                assert t_y is not None
                t_hit, hit_x_wall = t_y, False

            decay = math.exp(-BALL_DRAG * t_hit)
            factor = (1.0 - decay) / BALL_DRAG
            new_x = x + vx * factor
            new_y = y + vy * factor

            if hit_x_wall:
                new_vx, new_vy = -vx * decay, vy * decay
            else:
                new_vx, new_vy = vx * decay, -vy * decay

            t_cum += t_hit
            x, y = new_x, new_y
            vx, vy = new_vx, new_vy
            path.append(BallPathPoint((x, y), (vx, vy), t_cum))

        # Safety fallback: ball still moving after max_bounces — append stop estimate.
        speed = math.hypot(vx, vy)
        if speed >= _BALL_STOP_SPEED:
            t_stop = math.log(speed / _BALL_STOP_SPEED) / BALL_DRAG
            decay = math.exp(-BALL_DRAG * t_stop)
            factor = (1.0 - decay) / BALL_DRAG
            path.append(
                BallPathPoint(
                    (x + vx * factor, y + vy * factor),
                    (0.0, 0.0),
                    t_cum + t_stop,
                )
            )

        return path

    def closest_player(self, players: list[PlayerState]) -> PlayerState:
        """Return the player in ``players`` closest to the ball."""
        return min(players, key=lambda p: p.dist_to(self.location))

    # ── Interception helpers ───────────────────────────────────────────────

    def interception_times(
        self, players: list[PlayerState]
    ) -> list[tuple[float, PlayerState]]:
        """Return ``players`` sorted by how quickly each can intercept the ball.

        Each entry is ``(intercept_time_seconds, player)``.  The list is sorted
        ascending so ``result[0]`` gives the fastest interceptor.

        Useful for computing arrival order, intercept advantage, or any logic
        that needs to rank players by their ability to reach the ball.
        """
        return sorted(
            ((p.intercept_time(self), p) for p in players),
            key=lambda tp: tp[0],
        )

    def fastest_interceptor(self, players: list[PlayerState]) -> PlayerState:
        """Return the player who can intercept the ball in the least time."""
        return self.interception_times(players)[0][1]

    def intercept_advantage(
        self, player: PlayerState, opponents: list[PlayerState]
    ) -> float:
        """Seconds by which ``player`` beats the nearest opponent to the ball.

        A positive value means ``player`` arrives first; negative means at
        least one opponent is faster.  Returns ``math.inf`` when ``opponents``
        is empty (no competition).
        """
        if not opponents:
            return math.inf
        player_time = player.intercept_time(self)
        nearest_opponent_time = self.interception_times(opponents)[0][0]
        return nearest_opponent_time - player_time

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
        is_home: bool = True,
    ) -> None:
        self.team = team
        self.opposition = opposition
        self.ball = ball
        self.match_state = match_state
        self.is_home = is_home

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

    @property
    def opposition_has_ball(self) -> bool:
        """``True`` if any opposition currently holds the ball."""
        return any(p.has_ball for p in self.opposition)

    @property
    def loose_ball(self) -> bool:
        """``True`` if the ball is not currently being held by any player."""
        return not self.team_has_ball and not self.opposition_has_ball

    @property
    def global_ball_location(self) -> tuple[float, float]:
        """Ball position in the shared global frame (home team attacks right).

        Equivalent to ``ball.location`` for the home team's ``GameState``; for
        the away team's ``GameState`` the coordinates are rotated 180° back to
        the global reference frame.
        """
        if self.is_home:
            return self.ball.location
        bx, by = self.ball.location
        return (STANDARD_PITCH_WIDTH - bx, STANDARD_PITCH_HEIGHT - by)

    @property
    def global_ball_velocity(self) -> tuple[float, float]:
        """Ball velocity in the shared global frame (home team attacks right)."""
        if self.is_home:
            return self.ball.velocity
        vx, vy = self.ball.velocity
        return (-vx, -vy)

    def team_player(self, number: int) -> PlayerState | None:
        """Return the teammate with the given shirt number, or ``None``."""
        return next((p for p in self.team if p.number == number), None)

    def opposition_player(self, number: int) -> PlayerState | None:
        """Return the opposition player with the given shirt number, or ``None``."""
        return next((p for p in self.opposition if p.number == number), None)

    def players(
        self,
        *,
        teammates: bool = True,
        opposition: bool = True,
        player_on_ball: bool = True,
        players_on_cooldown: bool = True,
        include_goalkeepers: bool = True,
        include_outfield: bool = True,
        n_goalkeepers: int = 1,
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
        include_goalkeepers:
            Include goalkeepers (default: True). Goalkeepers aren't mechanically
            distinct to outfield players. This is a convenience parameter to make it
            easier to model the protection of goal for AI. The player(s) closest to goal
            are treated as the team's goalkeepers.
        include_outfield:
            Include outfield players (default: True). Outfield players are all players
            that are not goalkeepers.
        n_goalkeepers:
            Number of goalkeepers to include in each team (default: 1).
        """
        _own_goal = (0.0, STANDARD_PITCH_HEIGHT / 2)
        _opp_goal = (STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT / 2)
        team_goalkeepers: list[PlayerState] = sorted(
            self.team, key=lambda p: p.dist_to(_own_goal)
        )[:n_goalkeepers]
        opposition_goalkeepers: list[PlayerState] = sorted(
            self.opposition, key=lambda p: p.dist_to(_opp_goal)
        )[:n_goalkeepers]
        goalkeepers: list[PlayerState] = team_goalkeepers + opposition_goalkeepers
        outfield: list[PlayerState] = [
            p for p in self.all_players if p not in goalkeepers
        ]
        pool: list[PlayerState] = []
        if teammates:
            pool.extend(self.team)
        if opposition:
            pool.extend(self.opposition)
        if not player_on_ball:
            pool = [p for p in pool if not p.has_ball]
        if not players_on_cooldown:
            pool = [p for p in pool if p.cooldown_timer == 0.0]
        if not include_goalkeepers:
            pool = [p for p in pool if p not in goalkeepers]
        if not include_outfield:
            pool = [p for p in pool if p not in outfield]
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
