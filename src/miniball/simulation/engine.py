"""Pure-Python game simulation – no rendering or input dependencies.

All positions and velocities are stored in **global coordinates**
(x ∈ [0, 120], y ∈ [0, 80]) — the team-agnostic normalised pitch
frame.  The UI layer (``miniball.ui``) is responsible for converting to
screen pixels for rendering and for translating raw input into global
coordinates before passing it to the simulation.

Can be driven by an arcade window (``FootballGame`` in ``ui/game.py``) for the
interactive game, or run headlessly for batch analysis and AI league matches::

    sim = MatchSimulation(team_a_config, team_b_config)
    dt = 1 / 60          # fixed time-step; runs as fast as the CPU allows
    while not sim.game_over:
        sim.step(dt)
    df = sim.build_match_df()

Human input is injected through ``HumanInput`` objects passed to
``MatchSimulation.step()``.  Passing ``None`` (the default) produces a
fully AI-driven simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import polars as pl

from miniball.ai import (
    BallState,
    BaseAI,
    GameState,
    MatchState,
    PlayerState,
    TeamActions,
)
from miniball.config import (
    BALL_DRAG,
    BALL_RADIUS,
    GAME_DURATION,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    STANDARD_GOAL_DEPTH,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    STRIKE_ANGULAR_ERROR_DEGREES,
    STRIKE_COOLDOWN,
    STRIKE_SPEED,
    TACKLE_COOLDOWN,
)
from miniball.coords import (
    global_delta_to_team,
    global_to_team,
    team_delta_to_global,
    team_to_global,
)
from miniball.simulation.recording import FrameRecord
from miniball.teams import Team
from miniball.ui.config import C_TEAM_A, C_TEAM_B

# ── Public data types ─────────────────────────────────────────────────────────


@dataclass
class HumanInput:
    """One frame of human input to inject into the simulation.

    ``direction`` is in *global* coordinates (right = +x, up = +y) — the
    same frame as keyboard / gamepad screen-space input.  The simulation
    converts it to the appropriate team frame internally.
    """

    is_home: bool
    player_number: int
    direction: tuple[float, float]
    strike: bool


# ── Entities ──────────────────────────────────────────────────────────────────


class Ball:
    """Ball entity with position and velocity in global coordinates."""

    _CENTRE_X = STANDARD_PITCH_WIDTH / 2
    _CENTRE_Y = STANDARD_PITCH_HEIGHT / 2

    def __init__(self) -> None:
        self.x = self._CENTRE_X
        self.y = self._CENTRE_Y
        self.vx = 0.0
        self.vy = 0.0
        self.possessed_by: Player | None = None

    def reset(self) -> None:
        self.x = self._CENTRE_X
        self.y = self._CENTRE_Y
        self.vx = 0.0
        self.vy = 0.0
        self.possessed_by = None

    def apply_impulse(self, ix: float, iy: float) -> None:
        self.vx += ix
        self.vy += iy

    def update(self, dt: float) -> None:
        if self.possessed_by is not None:
            p = self.possessed_by
            self.x = p.x + math.cos(p.facing) * PLAYER_RADIUS
            self.y = p.y + math.sin(p.facing) * PLAYER_RADIUS
            self.vx = 0.0
            self.vy = 0.0
            return

        self.x += self.vx * dt
        self.y += self.vy * dt

        drag_factor = max(0.0, 1.0 - BALL_DRAG * dt)
        self.vx *= drag_factor
        self.vy *= drag_factor

        goal_lo = self._CENTRE_Y - STANDARD_GOAL_HEIGHT / 2
        goal_hi = self._CENTRE_Y + STANDARD_GOAL_HEIGHT / 2

        if self.y - BALL_RADIUS < 0:
            self.y = BALL_RADIUS
            self.vy = abs(self.vy)
        if self.y + BALL_RADIUS > STANDARD_PITCH_HEIGHT:
            self.y = STANDARD_PITCH_HEIGHT - BALL_RADIUS
            self.vy = -abs(self.vy)

        if self.x - BALL_RADIUS < 0 and not (goal_lo <= self.y <= goal_hi):
            self.x = BALL_RADIUS
            self.vx = abs(self.vx)
        if self.x + BALL_RADIUS > STANDARD_PITCH_WIDTH and not (
            goal_lo <= self.y <= goal_hi
        ):
            self.x = STANDARD_PITCH_WIDTH - BALL_RADIUS
            self.vx = -abs(self.vx)

        if self.x - BALL_RADIUS < -STANDARD_GOAL_DEPTH and goal_lo <= self.y <= goal_hi:
            self.x = -STANDARD_GOAL_DEPTH + BALL_RADIUS
            self.vx = abs(self.vx)
        if (
            self.x + BALL_RADIUS > STANDARD_PITCH_WIDTH + STANDARD_GOAL_DEPTH
            and goal_lo <= self.y <= goal_hi
        ):
            self.x = STANDARD_PITCH_WIDTH + STANDARD_GOAL_DEPTH - BALL_RADIUS
            self.vx = -abs(self.vx)


class Player:
    def __init__(
        self,
        number: int,
        x: float,
        y: float,
        color: tuple[int, int, int],
        is_home: bool,
    ) -> None:
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.color = color
        self.number = number
        self.is_home = is_home
        self.facing: float = 0.0 if is_home else math.pi
        # Single cooldown timer – blocks ball possession regains only;
        # movement is always permitted.  Set to TACKLE_COOLDOWN after a
        # tackle or STRIKE_COOLDOWN after striking (whichever is larger).
        self.cooldown_timer: float = 0.0

    @property
    def on_cooldown(self) -> bool:
        return self.cooldown_timer > 0

    @property
    def can_gain_possession(self) -> bool:
        return self.cooldown_timer <= 0

    def tick(self, dt: float) -> None:
        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

    def reset(self) -> None:
        self.x = self.start_x
        self.y = self.start_y
        self.facing = 0.0 if self.is_home else math.pi
        self.cooldown_timer = 0.0


# ── Simulation ────────────────────────────────────────────────────────────────


class MatchSimulation:
    """Self-contained match simulation with no arcade / rendering dependency.

    Parameters
    ----------
    team_a_config, team_b_config:
        Team definitions (AI, formation, name).  Team A is the home team and
        always attacks left → right (positive x direction).
    """

    def __init__(
        self,
        team_a_config: Team,
        team_b_config: Team,
    ) -> None:
        self.team_a_config = team_a_config
        self.team_b_config = team_b_config
        self._ai_a: BaseAI = team_a_config.ai
        self._ai_b: BaseAI = team_b_config.ai

        self.ball = Ball()
        self.team_a: list[Player] = []
        self.team_b: list[Player] = []
        self.score_a = 0
        self.score_b = 0
        self._time_remaining: float = GAME_DURATION
        self._game_over = False
        self._goal_flash: float = 0.0
        self._countdown: float = 3.0  # pre-kick-off / post-goal freeze

        self._history: list[FrameRecord] = []

        for player_config in team_a_config.players:
            gx, gy = team_to_global(player_config.x / 2, player_config.y, is_home=True)
            self.team_a.append(
                Player(
                    x=gx,
                    y=gy,
                    color=C_TEAM_A,
                    number=player_config.number,
                    is_home=True,
                )
            )
        for player_config in team_b_config.players:
            gx, gy = team_to_global(player_config.x / 2, player_config.y, is_home=False)
            self.team_b.append(
                Player(
                    x=gx,
                    y=gy,
                    color=C_TEAM_B,
                    number=player_config.number,
                    is_home=False,
                )
            )

        # Home team (attacks left→right) always takes the opening kick-off.
        self._assign_kickoff_possession(self.team_a)

    # ── Public read-only properties ───────────────────────────────────────────

    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def goal_flash(self) -> float:
        """Seconds remaining on the post-goal flash timer (0 when inactive)."""
        return self._goal_flash

    @property
    def countdown(self) -> float:
        """Seconds remaining on the kick-off freeze countdown."""
        return self._countdown

    @property
    def time_remaining(self) -> float:
        return self._time_remaining

    @property
    def all_players(self) -> list[Player]:
        return self.team_a + self.team_b

    # ── Main loop ─────────────────────────────────────────────────────────────

    def step(self, dt: float, human_input: HumanInput | None = None) -> None:
        """Advance the simulation by ``dt`` seconds.

        Parameters
        ----------
        dt:
            Time delta in seconds.  Values above 1/30 are clamped so that a
            stall (e.g. debugger pause) doesn't cause a physics explosion.
        human_input:
            Optional human override for one player this frame.  Pass ``None``
            for fully AI-driven simulation.
        """
        dt = min(dt, 1 / 30)

        if self._game_over:
            return

        if self._goal_flash > 0:
            self._goal_flash -= dt
            if self._goal_flash <= 0:
                self._goal_flash = 0.0
                self._countdown = 3.0
            return

        if self._countdown > 0:
            self._countdown = max(0.0, self._countdown - dt)
            return

        # ── Time ──────────────────────────────────────────────────────────────
        self._time_remaining -= dt
        if self._time_remaining <= 0:
            self._time_remaining = 0.0
            self._game_over = True
            self.ball.possessed_by = None
            return

        # 1. Tick all player timers
        for p in self.all_players:
            p.tick(dt)

        # 2. Loose-ball pickup: assign possession to the closest eligible player
        #    before AI decisions are made.  This ensures the receiving player sees
        #    themselves holding the ball when their AI runs in step 3, giving them
        #    a chance to act (pass, clear) before any tackle check.
        self._pickup_loose_ball()

        # 3. Build game states and compute effective actions (AI + human override).
        home_team_state = self._build_game_state(True)
        away_team_state = self._build_game_state(False)
        effective_a = self._get_team_actions(
            self._ai_a, home_team_state, is_home_team=True, human_input=human_input
        )
        effective_b = self._get_team_actions(
            self._ai_b, away_team_state, is_home_team=False, human_input=human_input
        )

        # 4. Record this frame for post-game analytics.
        self._history.append(
            FrameRecord(
                state=home_team_state,
                actions_team_a=effective_a,
                actions_team_b=effective_b,
                human_player=(
                    (human_input.is_home, human_input.player_number)
                    if human_input is not None
                    else None
                ),
            )
        )

        # 5. Apply effective actions; team vectors are converted from the team
        #    frame to global coordinates inside _apply_actions.
        self._apply_actions(self.team_a, effective_a, dt, is_home=True)
        self._apply_actions(self.team_b, effective_b, dt, is_home=False)

        # 6. Clamp players, resolve tackles, resolve collisions, clamp again.
        #    Tackle check runs after actions so the possessor's strike (step 5)
        #    pre-empts the tackle: if they passed, possessed_by is already None.
        self._clamp_players()
        self._resolve_tackles()
        self._resolve_player_collisions()
        self._clamp_players()

        # 7. Update ball (tracks possessor or advances with physics).
        self.ball.update(dt)

        # 8. Goal detection.
        self._check_goals()

    def simulate_match(self, dt: float = 1 / 60) -> pl.DataFrame | None:
        """Simulate a match and return the match DataFrame."""
        while not self.game_over:
            self.step(dt)
        return self.build_match_df()

    # ── Analytics ─────────────────────────────────────────────────────────────

    def export_history(self) -> None:
        """Write the match history to a parquet file on demand."""
        from miniball.simulation.recording import write_parquet

        df = self.build_match_df()
        if df is not None:
            write_parquet(df, self.team_a_config.name, self.team_b_config.name)

    # ── AI interface ──────────────────────────────────────────────────────────

    def _get_team_actions(
        self,
        ai: BaseAI,
        state: GameState,
        *,
        is_home_team: bool,
        human_input: HumanInput | None,
    ) -> TeamActions:
        """Return effective ``TeamActions`` for one team this frame.

        Layer 1 – AI decisions
            ``ai.get_actions(state)`` produces base actions for every player.

        Layer 2 – human override
            If ``human_input`` targets this team, that player's entry is
            replaced with the human's direction (converted to the team frame)
            and strike flag.
        """
        actions = ai.get_actions(state)

        if human_input is not None and human_input.is_home == is_home_team:
            dx, dy = global_delta_to_team(
                human_input.direction[0],
                human_input.direction[1],
                is_home=is_home_team,
            )
            actions[human_input.player_number] = {
                "direction": (dx, dy),
                "strike": human_input.strike,
            }

        return actions

    def _build_game_state(self, perspective_team_is_home: bool) -> GameState:
        """Snapshot current game into a normalised AI state dict.

        All coordinates are flipped for Team B so that every AI always sees
        its own team attacking left → right.
        """
        is_home = perspective_team_is_home

        def pos(x: float, y: float) -> tuple[float, float]:
            return global_to_team(x, y, is_home=is_home)

        team: list[PlayerState] = [
            PlayerState(
                number=p.number,
                is_teammate=True,
                is_home=perspective_team_is_home,
                has_ball=self.ball.possessed_by is p,
                cooldown_timer=p.cooldown_timer,
                location=pos(p.x, p.y),
            )
            for p in self.all_players
            if p.is_home == perspective_team_is_home
        ]
        opposition: list[PlayerState] = [
            PlayerState(
                number=p.number,
                is_teammate=False,
                is_home=not perspective_team_is_home,
                has_ball=self.ball.possessed_by is p,
                cooldown_timer=p.cooldown_timer,
                location=pos(p.x, p.y),
            )
            for p in self.all_players
            if p.is_home != perspective_team_is_home
        ]

        team_score = self.score_a if perspective_team_is_home else self.score_b
        oppo_score = self.score_b if perspective_team_is_home else self.score_a

        return GameState(
            team=team,
            opposition=opposition,
            ball=BallState(
                location=pos(self.ball.x, self.ball.y),
                velocity=global_delta_to_team(
                    self.ball.vx, self.ball.vy, is_home=is_home
                ),
            ),
            match_state=MatchState(
                team_current_score=team_score,
                opposition_current_score=oppo_score,
                match_time_seconds=GAME_DURATION - self._time_remaining,
            ),
            is_home=perspective_team_is_home,
        )

    def _apply_actions(
        self,
        players: list[Player],
        actions: TeamActions,
        dt: float,
        is_home: bool = True,
    ) -> None:
        """Move and optionally strike for each player according to actions.

        The ``direction`` field is treated as a desired displacement vector in
        standard pitch coordinates.  The engine clamps the actual movement to
        ``PLAYER_SPEED * dt``; vectors shorter than that are applied as-is so
        that players can make fine adjustments without overshooting.
        """
        for p in players:
            player_action = actions.get(p.number)
            if player_action is None:
                continue
            dx, dy = player_action["direction"]
            dx, dy = team_delta_to_global(dx, dy, is_home=is_home)
            magnitude = math.hypot(dx, dy)
            if magnitude > 0:
                max_dist = PLAYER_SPEED * dt
                scale = min(1.0, max_dist / magnitude)
                p.x += dx * scale
                p.y += dy * scale
                p.facing = math.atan2(dy, dx)
            if player_action["strike"]:
                self._handle_strike(p)

    # ── Physics helpers ───────────────────────────────────────────────────────

    def _handle_strike(self, player: Player) -> None:
        """Launch the ball in ``player``'s facing direction with angular error."""
        if self.ball.possessed_by is not player:
            return
        self.ball.possessed_by = None
        error_rad = math.radians(STRIKE_ANGULAR_ERROR_DEGREES)
        angle = player.facing + random.uniform(-error_rad, error_rad)
        separation = PLAYER_RADIUS + BALL_RADIUS + 0.24
        self.ball.x = player.x + math.cos(angle) * separation
        self.ball.y = player.y + math.sin(angle) * separation
        self.ball.vx = math.cos(angle) * STRIKE_SPEED
        self.ball.vy = math.sin(angle) * STRIKE_SPEED
        player.cooldown_timer = max(player.cooldown_timer, STRIKE_COOLDOWN)

    def _pickup_loose_ball(self) -> None:
        """Assign a loose ball to the closest eligible player within pickup range.

        Called *before* AI decisions so the receiving player can act (pass,
        clear) in the same frame they receive the ball.

        When multiple players are within ``PLAYER_RADIUS + BALL_RADIUS``, the
        one physically closest to the ball wins — no home/away ordering bias.
        """
        if self.ball.possessed_by is not None:
            return
        candidates = [
            p
            for p in self.all_players
            if p.can_gain_possession
            and math.hypot(self.ball.x - p.x, self.ball.y - p.y)
            < PLAYER_RADIUS + BALL_RADIUS
        ]
        if not candidates:
            return
        winner = min(
            candidates, key=lambda p: math.hypot(self.ball.x - p.x, self.ball.y - p.y)
        )
        self.ball.possessed_by = winner
        self.ball.vx = 0.0
        self.ball.vy = 0.0

    def _resolve_tackles(self) -> None:
        """Transfer possession when a defender is within tackle range of the possessor.

        Called *after* actions are applied so a strike (pass or clearance) in
        the same frame pre-empts the tackle: if the possessor released the ball,
        ``possessed_by`` is already ``None`` and this is a no-op.
        """
        possessor = self.ball.possessed_by
        if possessor is None:
            return
        for p in self.all_players:
            if p.is_home == possessor.is_home:
                continue
            if not p.can_gain_possession:
                continue
            dist = math.hypot(p.x - possessor.x, p.y - possessor.y)
            if dist < PLAYER_RADIUS * 2:
                old = possessor
                self.ball.possessed_by = p
                old.cooldown_timer = TACKLE_COOLDOWN
                break

    def _resolve_player_collisions(self) -> None:
        players = self.all_players
        min_dist = PLAYER_RADIUS * 2
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                dist = math.hypot(b.x - a.x, b.y - a.y)
                if 0 < dist < min_dist:
                    angle = math.atan2(b.y - a.y, b.x - a.x)
                    push = (min_dist - dist) / 2
                    a.x -= math.cos(angle) * push
                    a.y -= math.sin(angle) * push
                    b.x += math.cos(angle) * push
                    b.y += math.sin(angle) * push

    def _clamp_players(self) -> None:
        for p in self.all_players:
            p.x = max(PLAYER_RADIUS, min(STANDARD_PITCH_WIDTH - PLAYER_RADIUS, p.x))
            p.y = max(PLAYER_RADIUS, min(STANDARD_PITCH_HEIGHT - PLAYER_RADIUS, p.y))

    def _check_goals(self) -> None:
        goal_lo = STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2
        goal_hi = STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2

        if self.ball.x + BALL_RADIUS < 0 and goal_lo <= self.ball.y <= goal_hi:
            self.score_b += 1
            self._trigger_goal_reset(conceding_team=self.team_a)
        elif (
            self.ball.x - BALL_RADIUS > STANDARD_PITCH_WIDTH
            and goal_lo <= self.ball.y <= goal_hi
        ):
            self.score_a += 1
            self._trigger_goal_reset(conceding_team=self.team_b)

    def _trigger_goal_reset(self, conceding_team: list[Player]) -> None:
        self._goal_flash = 1.5
        self.ball.reset()
        for p in self.all_players:
            p.reset()
        self._assign_kickoff_possession(conceding_team)

    def _assign_kickoff_possession(self, team: list[Player]) -> None:
        """Give the ball to the furthest-forward player in *team* at kick-off.

        Home team attacks right (high x = forward); away team attacks left
        (low x = forward).  Ties are broken at random.
        """
        if team[0].is_home:
            forward_x = max(p.start_x for p in team)
        else:
            forward_x = min(p.start_x for p in team)
        candidates = [p for p in team if p.start_x == forward_x]
        chosen = random.choice(candidates)
        self.ball.possessed_by = chosen
        self.ball.update(0)

    # ── Analytics export ──────────────────────────────────────────────────────

    def build_match_df(self) -> pl.DataFrame | None:
        """Flatten ``_history`` into a Polars DataFrame and return it.

        Each row represents one player at one frame.

        Coordinate conventions
        ──────────────────────
        All positional and directional columns (``player_x``, ``player_y``,
        ``action_dx``, ``action_dy``, ``ball_x``, ``ball_y``, ``ball_vx``,
        ``ball_vy``) are in the **team's own normalised frame**: the team
        always attacks right, X ∈ [0, 120], Y ∈ [0, 80].  This convention
        is consistent across home and away teams and across multiple matches,
        making it suitable for cross-game analysis.

        To obtain global coordinates (home team attacks right), use:
        ``global_x = player_x if is_home else (120 − player_x)``
        or reconstruct the per-frame ``GameState`` objects via
        ``reconstruct_frames()`` and access ``player.global_location``.
        """
        if not self._history:
            return None

        from miniball.simulation.recording import build_rows

        return pl.DataFrame(
            build_rows(
                self._history,
                self.team_a_config.name,
                self.team_b_config.name,
            )
        )
