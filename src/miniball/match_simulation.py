"""Pure-Python game simulation – no rendering or input dependencies.

Can be driven by an arcade window (``FootballGame`` in ``game.py``) for the
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
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from miniball.ai import (
    BaseAI,
    GameState,
    MatchState,
    PlayerAction,
    PlayerState,
    TeamActions,
)
from miniball.config import (
    BALL_DRAG,
    BALL_RADIUS,
    C_TEAM_A,
    C_TEAM_B,
    GAME_DURATION,
    GOAL_DEPTH,
    GOAL_H,
    MAX_BALL_SPEED,
    PITCH_B,
    PITCH_CX,
    PITCH_CY,
    PITCH_L,
    PITCH_R,
    PITCH_T,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    STRIKE_COOLDOWN,
    STRIKE_SPEED,
    TACKLE_COOLDOWN,
)
from miniball.coordinate_transformations import (
    global_delta_to_team,
    screen_delta_to_team,
    screen_to_team,
    team_delta_to_global,
    team_to_screen,
)
from miniball.teams import Team

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


@dataclass
class FrameRecord:
    """Single-frame snapshot used for post-game analysis.

    ``state`` is expressed in team A's coordinate system (team A always
    attacks right), which serves as the global pitch reference frame.

    ``actions_team_a`` and ``actions_team_b`` are each in their own team's
    normalised frame (both teams attack right in their respective frames).
    To convert team B's directions to the global frame, negate both components.

    Human input overrides are already merged in: these are the *effective*
    actions that were sent to the game engine, not the raw AI outputs.
    """

    game_time: float  # seconds elapsed since kick-off
    state: GameState  # global reference frame (team A perspective)
    actions_team_a: TeamActions  # normalised; team A attacks right
    actions_team_b: TeamActions  # normalised; team B's own frame (also attacks right)
    human_player: tuple[bool, int] | None  # (is_home, player_number) or None


# ── Entities ──────────────────────────────────────────────────────────────────


class Ball:
    def __init__(self) -> None:
        self.x = PITCH_CX
        self.y = PITCH_CY
        self.vx = 0.0
        self.vy = 0.0
        self.possessed_by: Player | None = None

    def reset(self) -> None:
        self.x = PITCH_CX
        self.y = PITCH_CY
        self.vx = 0.0
        self.vy = 0.0
        self.possessed_by = None

    def apply_impulse(self, ix: float, iy: float) -> None:
        self.vx += ix
        self.vy += iy
        speed = math.hypot(self.vx, self.vy)
        if speed > MAX_BALL_SPEED:
            scale = MAX_BALL_SPEED / speed
            self.vx *= scale
            self.vy *= scale

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

        goal_lo = PITCH_CY - GOAL_H / 2
        goal_hi = PITCH_CY + GOAL_H / 2

        if self.y - BALL_RADIUS < PITCH_B:
            self.y = PITCH_B + BALL_RADIUS
            self.vy = abs(self.vy)
        if self.y + BALL_RADIUS > PITCH_T:
            self.y = PITCH_T - BALL_RADIUS
            self.vy = -abs(self.vy)

        if self.x - BALL_RADIUS < PITCH_L and not (goal_lo <= self.y <= goal_hi):
            self.x = PITCH_L + BALL_RADIUS
            self.vx = abs(self.vx)
        if self.x + BALL_RADIUS > PITCH_R and not (goal_lo <= self.y <= goal_hi):
            self.x = PITCH_R - BALL_RADIUS
            self.vx = -abs(self.vx)

        if self.x - BALL_RADIUS < PITCH_L - GOAL_DEPTH and goal_lo <= self.y <= goal_hi:
            self.x = PITCH_L - GOAL_DEPTH + BALL_RADIUS
            self.vx = abs(self.vx)
        if self.x + BALL_RADIUS > PITCH_R + GOAL_DEPTH and goal_lo <= self.y <= goal_hi:
            self.x = PITCH_R + GOAL_DEPTH - BALL_RADIUS
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
            sx, sy = team_to_screen(player_config.x / 2, player_config.y, is_home=True)
            self.team_a.append(
                Player(
                    x=sx,
                    y=sy,
                    color=C_TEAM_A,
                    number=player_config.number,
                    is_home=True,
                )
            )
        for player_config in team_b_config.players:
            sx, sy = team_to_screen(player_config.x / 2, player_config.y, is_home=False)
            self.team_b.append(
                Player(
                    x=sx,
                    y=sy,
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

        # 2. Build game states and compute effective actions (AI + human override).
        home_team_state = self._build_game_state(True)
        away_team_state = self._build_game_state(False)
        effective_a = self._get_team_actions(
            self._ai_a, home_team_state, is_home_team=True, human_input=human_input
        )
        effective_b = self._get_team_actions(
            self._ai_b, away_team_state, is_home_team=False, human_input=human_input
        )

        # 3. Record this frame for post-game analytics.
        self._history.append(
            FrameRecord(
                game_time=GAME_DURATION - self._time_remaining,
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

        # 4. Apply effective actions; team B vectors are flipped from normalised
        #    back to screen coordinates inside _apply_actions.
        self._apply_actions(self.team_a, effective_a, dt, is_home=True)
        self._apply_actions(self.team_b, effective_b, dt, is_home=False)

        # 5. Clamp all players to pitch, resolve collisions, clamp again.
        self._clamp_players()
        self._update_possession()
        self._resolve_player_collisions()
        self._clamp_players()

        # 6. Update ball (tracks possessor or advances with physics).
        self.ball.update(dt)

        # 7. Goal detection.
        self._check_goals()

    def simulate_match(self, dt: float = 1 / 60) -> pl.DataFrame | None:
        """Simulate a match and return the match DataFrame."""
        while not self.game_over:
            self.step(dt)
        return self.build_match_df()

    # ── Analytics ─────────────────────────────────────────────────────────────

    def export_history(self) -> None:
        """Write the match history to a parquet file on demand."""
        df = self.build_match_df()
        if df is not None:
            self._write_parquet(df)

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
            actions["actions"][human_input.player_number] = {
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

        def pos(x: float, y: float) -> list[float]:
            return list(screen_to_team(x, y, is_home=is_home))

        team: list[PlayerState] = [
            {
                "number": p.number,
                "is_teammate": True,
                "has_ball": self.ball.possessed_by is p,
                "cooldown_timer": p.cooldown_timer,
                "location": pos(p.x, p.y),
            }
            for p in self.all_players
            if p.is_home == perspective_team_is_home
        ]
        opposition: list[PlayerState] = [
            {
                "number": p.number,
                "is_teammate": False,
                "has_ball": self.ball.possessed_by is p,
                "cooldown_timer": p.cooldown_timer,
                "location": pos(p.x, p.y),
            }
            for p in self.all_players
            if p.is_home != perspective_team_is_home
        ]

        team_score = self.score_a if perspective_team_is_home else self.score_b
        oppo_score = self.score_b if perspective_team_is_home else self.score_a
        match_state: MatchState = {
            "team_current_score": team_score,
            "opposition_current_score": oppo_score,
            "match_time_seconds": GAME_DURATION - self._time_remaining,
        }

        return {
            "team": team,
            "opposition": opposition,
            "ball": {
                "location": pos(self.ball.x, self.ball.y),
                "velocity": list(
                    screen_delta_to_team(self.ball.vx, self.ball.vy, is_home=is_home)
                ),
            },
            "match_state": match_state,
        }

    def _apply_actions(
        self,
        players: list[Player],
        actions: TeamActions,
        dt: float,
        is_home: bool = True,
    ) -> None:
        """Move and optionally strike for each player according to actions."""
        for p in players:
            player_action = actions["actions"].get(p.number)
            if player_action is None:
                continue
            dx, dy = player_action["direction"]
            dx, dy = team_delta_to_global(dx, dy, is_home=is_home)
            if dx != 0 or dy != 0:
                norm = math.hypot(dx, dy)
                speed_frac = min(norm, 1.0)
                p.x += (dx / norm) * PLAYER_SPEED * speed_frac * dt
                p.y += (dy / norm) * PLAYER_SPEED * speed_frac * dt
                p.facing = math.atan2(dy, dx)
            if player_action["strike"]:
                self._handle_strike(p)

    # ── Physics helpers ───────────────────────────────────────────────────────

    def _handle_strike(self, player: Player) -> None:
        """Launch the ball in ``player``'s facing direction."""
        if self.ball.possessed_by is not player:
            return
        self.ball.possessed_by = None
        self.ball.x = player.x + math.cos(player.facing) * (
            PLAYER_RADIUS + BALL_RADIUS + 2
        )
        self.ball.y = player.y + math.sin(player.facing) * (
            PLAYER_RADIUS + BALL_RADIUS + 2
        )
        self.ball.vx = math.cos(player.facing) * STRIKE_SPEED
        self.ball.vy = math.sin(player.facing) * STRIKE_SPEED
        player.cooldown_timer = max(player.cooldown_timer, STRIKE_COOLDOWN)

    def _update_possession(self) -> None:
        possessor = self.ball.possessed_by

        if possessor is not None:
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
        else:
            for p in self.all_players:
                if not p.can_gain_possession:
                    continue
                dist = math.hypot(self.ball.x - p.x, self.ball.y - p.y)
                if dist < PLAYER_RADIUS + BALL_RADIUS:
                    self.ball.possessed_by = p
                    self.ball.vx = 0.0
                    self.ball.vy = 0.0
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
            p.x = max(PITCH_L + PLAYER_RADIUS, min(PITCH_R - PLAYER_RADIUS, p.x))
            p.y = max(PITCH_B + PLAYER_RADIUS, min(PITCH_T - PLAYER_RADIUS, p.y))

    def _check_goals(self) -> None:
        goal_lo = PITCH_CY - GOAL_H / 2
        goal_hi = PITCH_CY + GOAL_H / 2

        if self.ball.x + BALL_RADIUS < PITCH_L and goal_lo <= self.ball.y <= goal_hi:
            self.score_b += 1
            self._trigger_goal_reset(conceding_team=self.team_a)
        elif self.ball.x - BALL_RADIUS > PITCH_R and goal_lo <= self.ball.y <= goal_hi:
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
        ``pos_x`` / ``pos_y`` / ``action_dx`` / ``action_dy`` are in the
        team's own normalised frame (the team always attacks right, x ∈ [0,
        120], y ∈ [0, 80]).  The ``_global`` variants are in the shared pitch
        frame where team A always attacks right; for team A these are identical,
        while for team B they are the 180° rotation (W − x, H − y).
        """
        if not self._history:
            return None

        from miniball.coordinate_transformations import (
            global_to_team,
            team_delta_to_global,
        )

        name_a = self.team_a_config.name
        name_b = self.team_b_config.name

        rows: list[dict[str, object]] = []
        for frame_number, record in enumerate(self._history):
            gbx, gby = record.state["ball"]["location"]
            gbvx, gbvy = record.state["ball"]["velocity"]
            score_a = record.state["match_state"]["team_current_score"]
            score_b = record.state["match_state"]["opposition_current_score"]
            match_time = record.state["match_state"]["match_time_seconds"]
            t = record.game_time

            _null_action: PlayerAction = {"direction": (0.0, 0.0), "strike": False}
            hp = record.human_player

            for player in record.state["team"]:  # team A – own frame = global
                num = player["number"]
                gx, gy = player["location"]
                pa_a = record.actions_team_a["actions"].get(num, _null_action)
                dx, dy = pa_a["direction"]
                rows.append(
                    {
                        "frame_number": frame_number,
                        "game_time": t,
                        "match_time_seconds": match_time,
                        "team": name_a,
                        "is_home": True,
                        "player_number": num,
                        "is_human_controlled": hp is not None
                        and hp[0] is True
                        and hp[1] == num,
                        "pos_x": gx,
                        "pos_y": gy,
                        "pos_x_global": gx,
                        "pos_y_global": gy,
                        "has_ball": player["has_ball"],
                        "cooldown_timer": player["cooldown_timer"],
                        "action_dx": dx,
                        "action_dy": dy,
                        "action_dx_global": dx,
                        "action_dy_global": dy,
                        "strike": pa_a["strike"],
                        "ball_x": gbx,
                        "ball_y": gby,
                        "ball_x_global": gbx,
                        "ball_y_global": gby,
                        "ball_vx": gbvx,
                        "ball_vy": gbvy,
                        "ball_vx_global": gbvx,
                        "ball_vy_global": gbvy,
                        "team_score": score_a,
                        "opposition_score": score_b,
                    }
                )

            for player in record.state["opposition"]:  # team B
                num = player["number"]
                gx, gy = player["location"]
                bx, by = global_to_team(gx, gy, is_home=False)
                pa_b = record.actions_team_b["actions"].get(num, _null_action)
                dx_b, dy_b = pa_b["direction"]
                adx_g, ady_g = team_delta_to_global(dx_b, dy_b, is_home=False)
                bbx, bby = global_to_team(gbx, gby, is_home=False)
                bbvx, bbvy = global_delta_to_team(gbvx, gbvy, is_home=False)
                rows.append(
                    {
                        "frame_number": frame_number,
                        "game_time": t,
                        "match_time_seconds": match_time,
                        "team": name_b,
                        "is_home": False,
                        "player_number": num,
                        "is_human_controlled": hp is not None
                        and hp[0] is False
                        and hp[1] == num,
                        "pos_x": bx,
                        "pos_y": by,
                        "pos_x_global": gx,
                        "pos_y_global": gy,
                        "has_ball": player["has_ball"],
                        "cooldown_timer": player["cooldown_timer"],
                        "action_dx": dx_b,
                        "action_dy": dy_b,
                        "action_dx_global": adx_g,
                        "action_dy_global": ady_g,
                        "strike": pa_b["strike"],
                        "ball_x": bbx,
                        "ball_y": bby,
                        "ball_x_global": gbx,
                        "ball_y_global": gby,
                        "ball_vx": bbvx,
                        "ball_vy": bbvy,
                        "ball_vx_global": gbvx,
                        "ball_vy_global": gbvy,
                        "team_score": score_b,
                        "opposition_score": score_a,
                    }
                )

        return pl.DataFrame(rows)

    def _write_parquet(self, df: pl.DataFrame) -> None:
        """Write a match history DataFrame to a timestamped parquet file."""
        out_dir = Path("match_data")
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        path = out_dir / f"match_{timestamp}_{unique_id}.parquet"
        df.write_parquet(path)
        print(
            f"Match data saved → {path}  ({len(self._history)} frames · {len(df)} rows)"
        )


if __name__ == "__main__":
    from miniball.teams import teams

    sim = MatchSimulation(teams["Baseline (1-2-2)"], teams["Baseline (1-3-1)"])
    df = sim.simulate_match()
    sim.export_history()
