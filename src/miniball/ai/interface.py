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
from typing import NamedTuple, NotRequired, TypedDict

from miniball.config import (
    BALL_DECEL,
    BALL_RADIUS,
    PLAYER_SPEED,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)

_PLAYER_SPEED_SQ: float = PLAYER_SPEED * PLAYER_SPEED

# Effective pitch boundaries for the ball centre (keeps ball_radius from walls)
_WALL_LEFT: float = BALL_RADIUS
_WALL_RIGHT: float = STANDARD_PITCH_WIDTH - BALL_RADIUS
_WALL_BOTTOM: float = BALL_RADIUS
_WALL_TOP: float = STANDARD_PITCH_HEIGHT - BALL_RADIUS

# Goal opening: y range of the goal mouth (same for both ends of the pitch)
_GOAL_LO: float = STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2
_GOAL_HI: float = STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2


def _time_to_wall(vel: float, speed: float, delta: float) -> float | None:
    """Time for the ball to travel *delta* units under constant deceleration.

    *vel* is the velocity component in the direction of *delta* (same sign);
    *speed* is the current ball speed (magnitude of the full velocity vector);
    *delta* is the signed displacement to the target (``wall - pos``).

    Returns ``None`` when the ball decelerates to rest before covering *delta*.

    Under linear decay the ball's position satisfies:
        pos(t) = pos₀ + vel·t − (vel/speed)·(BALL_DECEL/2)·t²

    Setting pos(t) = wall gives the quadratic:
        A·t² + B·t + C = 0
        A = BALL_DECEL·vel / (2·speed)
        B = −vel
        C = delta

    The smaller positive root is the first crossing time.
    """
    # Maximum displacement in this direction before the ball stops.
    max_disp = vel * speed / (2.0 * BALL_DECEL)
    if abs(delta) > abs(max_disp):
        return None  # ball decelerates to rest before reaching the target

    A = BALL_DECEL * vel / (2.0 * speed)
    B = -vel
    C = delta
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return (
            None  # floating-point safety; should not occur when max_disp check passed
        )
    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)
    positives = [t for t in (t1, t2) if t >= 0.0]
    return min(positives) if positives else None


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

    The ball moves under linear deceleration within the segment:

        ball(t) = seg_loc + seg_vel · (t − c·t²)
        c = BALL_DECEL / (2 · seg_speed)

    The interception condition is:

        |ball(t) − player|² = PLAYER_SPEED² · (seg_t0 + t)²

    Substituting ball(t) gives a quartic in t.  The constant-velocity
    quadratic (c=0) provides an initial estimate, then Newton–Raphson
    iterations refine it to the exact solution:

        f(t)  = |ball(t) − player|² − Vp²·(seg_t0+t)²
        f′(t) = 2·(ball(t)−player)·V·(1−2·c·t) − 2·Vp²·(seg_t0+t)

    Returns ``None`` when no non-negative solution exists.
    """
    dx = seg_lx - px
    dy = seg_ly - py

    # ── Step 1: constant-velocity quadratic for initial estimate ──────────
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
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-B - sqrt_disc) / (2.0 * A)
        t2 = (-B + sqrt_disc) / (2.0 * A)
        positives = [t for t in (t1, t2) if t >= 0.0]
        if not positives:
            return None
        t = min(positives)

    # ── Step 2: Newton–Raphson refinement for exact linear-decay solution ─
    seg_speed = math.hypot(seg_vx, seg_vy)
    if seg_speed > 1e-9:
        c = BALL_DECEL / (2.0 * seg_speed)
        for _ in range(8):
            u = t - c * t * t  # displacement factor: t_eff for this segment
            ball_dx = dx + seg_vx * u
            ball_dy = dy + seg_vy * u
            T = seg_t0 + t
            f = ball_dx * ball_dx + ball_dy * ball_dy - _PLAYER_SPEED_SQ * T * T
            du_dt = 1.0 - 2.0 * c * t
            fp = (
                2.0 * (ball_dx * seg_vx + ball_dy * seg_vy) * du_dt
                - 2.0 * _PLAYER_SPEED_SQ * T
            )
            if abs(fp) < 1e-12:
                break
            dt = f / fp
            t -= dt
            if abs(dt) < 1e-10:
                break

    return t if t >= 0.0 else None


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

        Iterates over segments of the ball's traced path (see
        :meth:`BallState.trace_path`).  Within each segment the ball is
        approximated as moving at the segment's initial velocity (constant-
        velocity assumption), giving the quadratic intercept equation:

            A·t² + B·t + C = 0
            A = |V|²  − PLAYER_SPEED²
            B = 2·(d·V − PLAYER_SPEED²·t₀)   (d = seg_loc − player, t₀ = seg start time)
            C = |d|²  − PLAYER_SPEED²·t₀²

        The first segment whose solution lies within the segment's duration
        determines the intercept.  If the ball has stopped before the player
        arrives, the time to walk to the rest position is returned instead.
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
                # Return the actual linear-decay ball position at t_seg, not the
                # constant-velocity projection used to solve for t_seg.
                seg_speed = math.hypot(vx, vy)
                if seg_speed > 1e-9:
                    f = t_seg - BALL_DECEL * t_seg * t_seg / (2.0 * seg_speed)
                    return (lx + vx * f, ly + vy * f)
                return (lx, ly)

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

        Uses the game engine's linear-deceleration model:

            speed(t) = max(0, speed₀ − BALL_DECEL · t)
            pos(t)   = pos₀ + vel₀ · (t_eff − BALL_DECEL · t_eff² / (2 · speed₀))

        where ``t_eff = min(t, t_stop)`` and ``t_stop = speed₀ / BALL_DECEL``.

        Ignores pitch boundaries and possession changes.  Negative ``t``
        returns the current position unchanged.
        """
        if t <= 0.0:
            return self.location
        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed <= 0.0:
            return self.location
        t_stop = speed / BALL_DECEL
        t_eff = min(t, t_stop)
        factor = t_eff - BALL_DECEL * t_eff * t_eff / (2.0 * speed)
        return (
            self.location[0] + self.velocity[0] * factor,
            self.location[1] + self.velocity[1] * factor,
        )

    def position_when_crossing_x(self, x: float) -> tuple[float, float] | None:
        """Project the ball's position when it first crosses ``x``.

        Returns ``None`` if the ball is stationary in x, already at ``x``, not
        moving toward ``x``, or decelerates to rest before reaching ``x``.
        """
        vx = self.velocity[0]
        if abs(vx) < 1e-6:
            return None
        delta = x - self.location[0]
        if abs(delta) < 1e-6:
            return None
        if (vx > 0.0) != (delta > 0.0):
            return None  # ball moving away from x
        speed = math.hypot(vx, self.velocity[1])
        t = _time_to_wall(vx, speed, delta)
        if t is None:
            return None  # ball stops before reaching x
        return self.projected_position(t)

    def trace_path(self, max_bounces: int = 10) -> list[BallPathPoint]:
        """Trace the ball's full path under linear deceleration, including wall bounces.

        Returns a list of :class:`BallPathPoint` waypoints::

            [start, bounce₁, bounce₂, …, stop]

        * **start** — the ball's current position and velocity (``time = 0``).
        * **bounce** — position and *post-bounce* velocity at each wall collision.
        * **stop** — where the ball comes to rest exactly (speed reaches zero);
          velocity is ``(0.0, 0.0)``.

        Each waypoint's ``time`` is the cumulative seconds from *now*.  Wall-hit
        times are computed exactly via the quadratic formula (no approximation).

        The list always contains at least two entries (start + stop), even for
        a stationary ball (in that case both entries are the same point).

        Bounce reflection rules:
        * Left / right wall  → x-velocity is negated; deceleration continues.
        * Top / bottom wall  → y-velocity is negated; deceleration continues.
        * Goal openings      → no bounce; the trace stops and the final waypoint
          is placed at the ball's rest position (which will overshoot the goal
          line — this is intentional).

        *max_bounces* caps the number of wall reflections computed (default 10)
        as a safety limit; in practice a ball will decelerate to rest long
        before this.
        """
        x, y = self.location
        vx, vy = self.velocity
        t_cum = 0.0

        path: list[BallPathPoint] = [BallPathPoint((x, y), (vx, vy), t_cum)]

        for _ in range(max_bounces):
            speed = math.hypot(vx, vy)
            if speed < 1e-9:
                # Already stopped; path ends at current point.
                break

            # Exact stop time and rest position under constant deceleration.
            t_stop = speed / BALL_DECEL
            # rest = pos + vel * t_stop / 2  (average velocity over the deceleration)
            rest_x = x + vx * t_stop / 2.0
            rest_y = y + vy * t_stop / 2.0

            if (
                _WALL_LEFT <= rest_x <= _WALL_RIGHT
                and _WALL_BOTTOM <= rest_y <= _WALL_TOP
            ):
                # Ball decelerates to rest inside the pitch — append exact stop point.
                path.append(
                    BallPathPoint(
                        (rest_x, rest_y),
                        (0.0, 0.0),
                        t_cum + t_stop,
                    )
                )
                return path

            # Find time to the relevant wall in each axis (only when moving toward it).
            # _time_to_wall(vel, speed, delta) solves the quadratic for constant decel.
            t_x_raw = (
                _time_to_wall(vx, speed, (_WALL_RIGHT if vx > 0.0 else _WALL_LEFT) - x)
                if abs(vx) > 1e-9
                else None
            )
            t_y = (
                _time_to_wall(vy, speed, (_WALL_TOP if vy > 0.0 else _WALL_BOTTOM) - y)
                if abs(vy) > 1e-9
                else None
            )

            # Check whether the x-wall hit is actually a goal opening.
            # Compute the ball's y at the moment it reaches the left/right wall.
            # If that y falls in the goal mouth and the ball gets there before any
            # y-wall hit, stop tracing (no bounce off a goal); the fallback below
            # will append the rest point behind the goal line.
            # If a y-wall hit comes first, skip the x-wall this iteration so the
            # y-bounce is processed; the goal check repeats on the next pass.
            t_x = t_x_raw
            if t_x_raw is not None:
                factor_x = t_x_raw - BALL_DECEL * t_x_raw * t_x_raw / (2.0 * speed)
                y_at_x_wall = y + vy * factor_x
                if _GOAL_LO <= y_at_x_wall <= _GOAL_HI:
                    if t_y is None or t_x_raw <= t_y:
                        # Ball enters goal before any y-wall — stop tracing.
                        break
                    # y-wall hit comes first; defer the goal check to next pass.
                    t_x = None

            if t_x is None and t_y is None:
                # Neither wall reachable (shouldn't happen when rest is outside pitch).
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

            # Ball position and speed at the wall hit.
            factor = t_hit - BALL_DECEL * t_hit * t_hit / (2.0 * speed)
            new_x = x + vx * factor
            new_y = y + vy * factor
            v_ratio = (speed - BALL_DECEL * t_hit) / speed  # speed fraction remaining

            if hit_x_wall:
                new_vx, new_vy = -vx * v_ratio, vy * v_ratio
            else:
                new_vx, new_vy = vx * v_ratio, -vy * v_ratio

            t_cum += t_hit
            x, y = new_x, new_y
            vx, vy = new_vx, new_vy
            path.append(BallPathPoint((x, y), (vx, vy), t_cum))

        # Safety fallback: ball still moving after max_bounces — append exact stop.
        speed = math.hypot(vx, vy)
        if speed > 1e-9:
            t_stop = speed / BALL_DECEL
            path.append(
                BallPathPoint(
                    (x + vx * t_stop / 2.0, y + vy * t_stop / 2.0),
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
    strike_weight: NotRequired[
        float
    ]  # optional weight in [0.01, 1.0]; defaults to 0.5 if omitted


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
