"""Miniball – 5-a-side football built with the arcade library.

Top-down view, no sprites: players are coloured circles, the ball is a white
circle.  One player (yellow ring) is controlled with the arrow keys.

Possession model
────────────────
• When a free ball touches any player, that player absorbs it instantly.
• The ball sits at the possessor's front edge and travels with them.
• An opposition player who is NOT stunned and whose body overlaps the
  possessor's body steals possession immediately.
• The player who lost the ball is frozen for STUN_DURATION seconds
  (orange ring) and cannot gain possession while stunned.
• The controlled player shoots with Space / gamepad button A: the ball is
  launched in the direction they are facing, and they get a brief pickup
  cooldown so they don't immediately re-absorb their own shot.
• A gamepad's left analogue stick gives full 360° motion; keyboard arrow
  keys are still supported as a fallback.
"""

from __future__ import annotations

import math

import arcade

from miniball.ai import BaseAI, GameState, MatchState, PlayerState, TeamActions
from miniball.config import (
    BALL_DRAG,
    BALL_RADIUS,
    C_BALL,
    C_BALL_OUTLINE,
    C_CONTROLLED,
    C_GOAL,
    C_GRASS,
    C_HINT,
    C_HUD,
    C_LINE,
    C_PLAYER_OUTLINE,
    C_POSSESSION,
    C_TEAM_A,
    C_TEAM_B,
    COOLDOWN_ALPHA,
    GAME_DURATION,
    GOAL_DEPTH,
    GOAL_H,
    JOY_DEAD_ZONE,
    MAX_BALL_SPEED,
    PITCH_B,
    PITCH_CX,
    PITCH_CY,
    PITCH_L,
    PITCH_R,
    PITCH_T,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    SCREEN_H,
    SCREEN_W,
    SHOOT_SPEED,
    SHOT_COOLDOWN,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    TACKLE_COOLDOWN,
    TITLE,
)
from miniball.coordinate_transformations import (
    normalized_to_screen,
    screen_to_normalized,
)
from miniball.team_config import TeamConfig

# ── Entities ─────────────────────────────────────────────────────────────────


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
            # Sit at the possessor's front edge
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

        # Top / bottom walls
        if self.y - BALL_RADIUS < PITCH_B:
            self.y = PITCH_B + BALL_RADIUS
            self.vy = abs(self.vy)
        if self.y + BALL_RADIUS > PITCH_T:
            self.y = PITCH_T - BALL_RADIUS
            self.vy = -abs(self.vy)

        # Side walls – pass through goal opening
        if self.x - BALL_RADIUS < PITCH_L and not (goal_lo <= self.y <= goal_hi):
            self.x = PITCH_L + BALL_RADIUS
            self.vx = abs(self.vx)
        if self.x + BALL_RADIUS > PITCH_R and not (goal_lo <= self.y <= goal_hi):
            self.x = PITCH_R - BALL_RADIUS
            self.vx = -abs(self.vx)

        # Back wall of each goal box
        if self.x - BALL_RADIUS < PITCH_L - GOAL_DEPTH and goal_lo <= self.y <= goal_hi:
            self.x = PITCH_L - GOAL_DEPTH + BALL_RADIUS
            self.vx = abs(self.vx)
        if self.x + BALL_RADIUS > PITCH_R + GOAL_DEPTH and goal_lo <= self.y <= goal_hi:
            self.x = PITCH_R + GOAL_DEPTH - BALL_RADIUS
            self.vx = -abs(self.vx)

    def draw(self) -> None:
        arcade.draw_circle_filled(self.x, self.y, BALL_RADIUS, C_BALL)
        arcade.draw_circle_outline(self.x, self.y, BALL_RADIUS, C_BALL_OUTLINE, 2)


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
        # Face toward the opposing half at kick-off
        self.facing: float = 0.0 if is_home else math.pi
        # Single cooldown timer – blocks ball possession regains only;
        # movement is always permitted.  Set to TACKLE_COOLDOWN after a
        # tackle or SHOT_COOLDOWN after shooting (whichever is larger).
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

    def draw(self, highlight: bool = False, has_ball: bool = False) -> None:
        # Cooldown players are faded to make them visually distinct
        if self.on_cooldown:
            r, g, b = self.color[:3]
            fill_color: tuple[int, ...] = (r, g, b, COOLDOWN_ALPHA)
            outline_color: tuple[int, ...] = (*C_PLAYER_OUTLINE, COOLDOWN_ALPHA)
            text_color: tuple[int, ...] = (255, 255, 255, COOLDOWN_ALPHA)
        else:
            fill_color = self.color
            outline_color = C_PLAYER_OUTLINE
            text_color = C_LINE

        # Possession ring drawn before body so body paints over the inner edge
        if has_ball:
            arcade.draw_circle_outline(
                self.x, self.y, PLAYER_RADIUS + 5, C_POSSESSION, 3
            )

        # Player body
        arcade.draw_circle_filled(self.x, self.y, PLAYER_RADIUS, fill_color)
        arcade.draw_circle_outline(self.x, self.y, PLAYER_RADIUS, outline_color, 2)

        # Yellow ring – keyboard-controlled player (drawn over body outline)
        if highlight:
            arcade.draw_circle_outline(
                self.x, self.y, PLAYER_RADIUS + 2, C_CONTROLLED, 2
            )

        # Jersey number
        arcade.draw_text(
            str(self.number),
            self.x,
            self.y,
            text_color,
            font_size=11,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )


# ── Main window ──────────────────────────────────────────────────────────────


class FootballGame(arcade.Window):
    def __init__(self, team_a_config: TeamConfig, team_b_config: TeamConfig) -> None:
        super().__init__(SCREEN_W, SCREEN_H, TITLE)
        arcade.set_background_color((30, 30, 30, 255))

        self.ball = Ball()
        self.team_a_config = team_a_config
        self.team_b_config = team_b_config
        self.team_a: list[Player] = []
        self.team_b: list[Player] = []
        self.score_a = 0
        self.score_b = 0
        self._time_remaining: float = GAME_DURATION
        self._game_over = False
        self._keys: set[int] = set()
        self._goal_flash = 0.0
        self._countdown: float = 3.0  # freeze before kick-off / after each goal

        # Gamepad – use the first connected controller if one is present
        joysticks = arcade.get_joysticks()
        self._joystick = joysticks[0] if joysticks else None
        if self._joystick is not None:
            self._joystick.open()
        self._joy_shoot_prev = False  # edge-detect the shoot button

        for player_config in team_a_config.players:
            sx, sy = normalized_to_screen(player_config.x / 2, player_config.y)
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
            # Team B attacks left, so their normalised positions are 180°-rotated
            sx, sy = normalized_to_screen(
                player_config.x / 2, player_config.y, flip=True
            )
            self.team_b.append(
                Player(
                    x=sx,
                    y=sy,
                    color=C_TEAM_B,
                    number=player_config.number,
                    is_home=False,
                )
            )

        self._ai_a: BaseAI = team_a_config.ai
        self._ai_b: BaseAI = team_b_config.ai

    @property
    def _controlled(self) -> Player | None:
        """Human-controlled player, or ``None`` when the team is fully AI-driven."""
        idx = self.team_a_config.human_controlled
        return self.team_a[idx] if idx is not None else None

    @property
    def _all_players(self) -> list[Player]:
        return self.team_a + self.team_b

    # ── Drawing ──────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        self._draw_pitch()
        self.ball.draw()
        for p in self._all_players:
            p.draw(
                highlight=(p is self._controlled),
                has_ball=(self.ball.possessed_by is p),
            )
        self._draw_hud()

    def _draw_pitch(self) -> None:
        lw = 2
        goal_lo = PITCH_CY - GOAL_H / 2
        goal_hi = PITCH_CY + GOAL_H / 2

        arcade.draw_lrbt_rectangle_filled(PITCH_L, PITCH_R, PITCH_B, PITCH_T, C_GRASS)
        arcade.draw_lrbt_rectangle_outline(
            PITCH_L, PITCH_R, PITCH_B, PITCH_T, C_LINE, lw
        )
        arcade.draw_line(PITCH_CX, PITCH_B, PITCH_CX, PITCH_T, C_LINE, lw)
        arcade.draw_circle_outline(PITCH_CX, PITCH_CY, 70, C_LINE, lw)
        arcade.draw_circle_filled(PITCH_CX, PITCH_CY, 4, C_LINE)

        pa_w, pa_h = 150, 260
        arcade.draw_lrbt_rectangle_outline(
            PITCH_L,
            PITCH_L + pa_w,
            PITCH_CY - pa_h / 2,
            PITCH_CY + pa_h / 2,
            C_LINE,
            lw,
        )
        arcade.draw_lrbt_rectangle_outline(
            PITCH_R - pa_w,
            PITCH_R,
            PITCH_CY - pa_h / 2,
            PITCH_CY + pa_h / 2,
            C_LINE,
            lw,
        )

        arcade.draw_lrbt_rectangle_filled(
            PITCH_L - GOAL_DEPTH, PITCH_L, goal_lo, goal_hi, C_GOAL
        )
        arcade.draw_lrbt_rectangle_outline(
            PITCH_L - GOAL_DEPTH, PITCH_L, goal_lo, goal_hi, C_LINE, lw
        )
        arcade.draw_lrbt_rectangle_filled(
            PITCH_R, PITCH_R + GOAL_DEPTH, goal_lo, goal_hi, C_GOAL
        )
        arcade.draw_lrbt_rectangle_outline(
            PITCH_R, PITCH_R + GOAL_DEPTH, goal_lo, goal_hi, C_LINE, lw
        )

    def _draw_hud(self) -> None:
        secs = max(0.0, self._time_remaining)
        mins = int(secs) // 60
        sec_part = int(secs) % 60
        timer_str = f"{mins}:{sec_part:02d}"
        arcade.draw_text(
            f"{self.team_a_config.name}  {self.score_a} – {self.score_b}  {self.team_b_config.name}",
            SCREEN_W / 2,
            SCREEN_H - 32,
            C_HUD,
            font_size=22,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        arcade.draw_text(
            f"{timer_str}",
            SCREEN_W - 20,
            SCREEN_H - 32,
            C_HUD,
            font_size=22,
            anchor_x="right",
            anchor_y="center",
            bold=True,
        )
        arcade.draw_text(
            "Arrows / L-stick to move and aim shot · Space / A to shoot",
            SCREEN_W / 2,
            28,
            C_HINT,
            font_size=12,
            anchor_x="center",
            anchor_y="center",
        )
        if self._game_over:
            arcade.draw_text(
                "FULL TIME",
                SCREEN_W / 2,
                SCREEN_H / 2,
                (255, 220, 0),
                font_size=72,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )
        elif self._goal_flash > 0:
            arcade.draw_text(
                "GOAL!",
                SCREEN_W / 2,
                SCREEN_H / 2,
                (255, 220, 0),
                font_size=72,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )
        elif self._countdown > 0:
            arcade.draw_text(
                str(math.ceil(self._countdown)),
                SCREEN_W / 2,
                SCREEN_H / 2,
                (255, 220, 0),
                font_size=120,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

    # ── Input ─────────────────────────────────────────────────────────────────

    def on_key_press(self, key: int, modifiers: int) -> None:
        self._keys.add(key)
        if key == arcade.key.SPACE:
            self._handle_shoot()

    def on_key_release(self, key: int, modifiers: int) -> None:
        self._keys.discard(key)

    # ── Update ────────────────────────────────────────────────────────────────

    def on_update(self, delta_time: float) -> None:
        dt = min(delta_time, 1 / 30)

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

        self._time_remaining -= dt
        if self._time_remaining <= 0:
            self._time_remaining = 0.0
            self._game_over = True
            self.ball.possessed_by = None
            return

        # 1. Tick all player timers
        for p in self._all_players:
            p.tick(dt)

        # 2. Move the controlled player (skipped when team is fully AI-driven)
        if self._controlled is not None:
            dx, dy = self._get_move_input()
            if dx != 0 or dy != 0:
                norm = math.hypot(dx, dy)
                self._controlled.x += (dx / norm) * PLAYER_SPEED * dt
                self._controlled.y += (dy / norm) * PLAYER_SPEED * dt
                self._controlled.facing = math.atan2(dy, dx)

            # Gamepad shoot button (A / Cross = button 0) – edge-triggered
            if self._joystick is not None and self._joystick.buttons:
                shoot_now = bool(self._joystick.buttons[0])
                if shoot_now and not self._joy_shoot_prev:
                    self._handle_shoot()
                self._joy_shoot_prev = shoot_now

        # 2b. AI move + shoot for non-human players.
        #     Each team's state is normalised to attack right; flip_x converts
        #     the returned move vectors back to screen coordinates.
        home_team_state = self._build_game_state(True)
        self._apply_ai_actions(
            [p for p in self.team_a if p is not self._controlled],
            self._ai_a.get_actions(home_team_state),
            dt,
            flip=False,  # Team A attacks right – no rotation needed
        )
        away_team_state = self._build_game_state(False)
        self._apply_ai_actions(
            [p for p in self.team_b if p is not self._controlled],
            self._ai_b.get_actions(away_team_state),
            dt,
            flip=True,  # Team B: 180° rotation back to screen coords
        )

        # 3. Clamp all players to pitch
        self._clamp_players()

        # 4. Possession transfer (checked BEFORE separation so overlaps are real)
        self._update_possession()

        # 5. Player–player positional separation
        self._resolve_player_collisions()
        self._clamp_players()

        # 6. Update ball (tracks possessor or advances with physics)
        self.ball.update(dt)

        # 7. Goal detection
        self._check_goals()

    # ── AI interface ─────────────────────────────────────────────────────────

    def _build_game_state(self, perspective_team_is_home: bool) -> GameState:
        """Snapshot the current game into a normalised AI state dict.

        All coordinates are flipped horizontally for Team B so that every AI
        always sees its team attacking left → right.  ``is_teammate`` is set
        to ``True`` for players on ``perspective_team``; engine-internal
        fields (``facing``, raw team label, attacking direction) are excluded.
        """
        flip = not perspective_team_is_home

        # screen_to_normalized converts pixel coords → standard pitch coords
        # (0–120 × 0–80) and, when flip=True, also applies the 180° rotation
        # that ensures both teams always see themselves attacking right.
        def pos(x: float, y: float) -> list[float]:
            nx, ny = screen_to_normalized(x, y, flip=flip)
            return [nx, ny]

        # Velocities have no positional offset: just scale and optionally negate.
        vel_sx = STANDARD_PITCH_WIDTH / SCREEN_W
        vel_sy = STANDARD_PITCH_HEIGHT / SCREEN_H
        sign = -1 if flip else 1

        team: list[PlayerState] = [
            {
                "number": p.number,
                "is_teammate": True,
                "has_ball": self.ball.possessed_by is p,
                "on_cooldown": p.on_cooldown,
                "location": pos(p.x, p.y),
            }
            for p in self._all_players
            if p.is_home == perspective_team_is_home
        ]
        opposition: list[PlayerState] = [
            {
                "number": p.number,
                "is_teammate": False,
                "has_ball": self.ball.possessed_by is p,
                "on_cooldown": p.on_cooldown,
                "location": pos(p.x, p.y),
            }
            for p in self._all_players
            if p.is_home != perspective_team_is_home
        ]

        team_score = self.score_a if perspective_team_is_home else self.score_b
        oppo_score = self.score_b if perspective_team_is_home else self.score_a
        match_state: MatchState = {
            "team_current_score": team_score,
            "opposition_current_score": oppo_score,
            "seconds_left": self._time_remaining,
        }

        return {
            "team": team,
            "opposition": opposition,
            "ball": {
                "location": pos(self.ball.x, self.ball.y),
                "velocity": [
                    self.ball.vx * vel_sx * sign,
                    self.ball.vy * vel_sy * sign,
                ],
            },
            "match_state": match_state,
        }

    def _apply_ai_actions(
        self,
        players: list[Player],
        actions: TeamActions,
        dt: float,
        flip: bool = False,
    ) -> None:
        """Move and optionally shoot for each player according to AI actions.

        Parameters
        ----------
        flip:
            When ``True`` both the X and Y components of each move vector are
            negated, converting from the AI's normalised (always-attack-right,
            180°-rotation-consistent) frame back into game screen coordinates.
            Pass ``True`` for Team B.
        """
        for p in players:
            direction = actions["directions"].get(p.number)
            if direction is None:
                continue
            dx, dy = direction
            if flip:
                dx, dy = -dx, -dy
            if dx != 0 or dy != 0:
                norm = math.hypot(dx, dy)
                # Allow sub-1 magnitude from analogue-style AI outputs
                speed_frac = min(norm, 1.0)
                p.x += (dx / norm) * PLAYER_SPEED * speed_frac * dt
                p.y += (dy / norm) * PLAYER_SPEED * speed_frac * dt
                p.facing = math.atan2(dy, dx)
            if actions["shoot"]:
                self._handle_shoot(p)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_move_input(self) -> tuple[float, float]:
        """Return a (dx, dy) vector from keyboard or gamepad left stick.

        Keyboard produces unit-axis values; the analogue stick returns raw
        floats in [-1, 1] so diagonal movement naturally scales to the true
        stick magnitude (full 360° at any speed up to PLAYER_SPEED).
        Gamepad input takes priority when the stick is outside the dead zone.
        """
        dx = dy = 0.0

        # Keyboard
        if arcade.key.LEFT in self._keys:
            dx -= 1.0
        if arcade.key.RIGHT in self._keys:
            dx += 1.0
        if arcade.key.DOWN in self._keys:
            dy -= 1.0
        if arcade.key.UP in self._keys:
            dy += 1.0

        # Gamepad left analogue stick (overrides keyboard when active)
        if self._joystick is not None:
            jx = self._joystick.x
            # Pyglet's Y axis is inverted relative to arcade's screen coordinates
            jy = -self._joystick.y
            if math.hypot(jx, jy) > JOY_DEAD_ZONE:
                dx, dy = jx, jy

        return dx, dy

    def _handle_shoot(self, player: Player | None = None) -> None:
        """Launch the ball in ``player``'s facing direction.

        Defaults to the keyboard-controlled player when called with no
        argument (e.g. from the Space-bar / gamepad-button handler).
        Silently does nothing if the given player doesn't have the ball,
        or if no human player is configured.
        """
        if player is None:
            player = self._controlled
        if player is None:
            return
        if self.ball.possessed_by is not player:
            return
        # Place ball just beyond the player's front edge so it isn't immediately re-absorbed
        self.ball.possessed_by = None
        self.ball.x = player.x + math.cos(player.facing) * (
            PLAYER_RADIUS + BALL_RADIUS + 2
        )
        self.ball.y = player.y + math.sin(player.facing) * (
            PLAYER_RADIUS + BALL_RADIUS + 2
        )
        self.ball.vx = math.cos(player.facing) * SHOOT_SPEED
        self.ball.vy = math.sin(player.facing) * SHOOT_SPEED
        player.cooldown_timer = max(player.cooldown_timer, SHOT_COOLDOWN)

    def _update_possession(self) -> None:
        possessor = self.ball.possessed_by

        if possessor is not None:
            # Any non-stunned opposition player whose body overlaps the possessor steals the ball
            for p in self._all_players:
                if p.is_home == possessor.is_home:
                    continue
                if not p.can_gain_possession:
                    continue
                dist = math.hypot(p.x - possessor.x, p.y - possessor.y)
                if dist < PLAYER_RADIUS * 2:
                    old = possessor
                    self.ball.possessed_by = p
                    old.cooldown_timer = TACKLE_COOLDOWN
                    break  # one tackle per frame
        else:
            # First eligible player whose body overlaps the ball absorbs it
            for p in self._all_players:
                if not p.can_gain_possession:
                    continue
                dist = math.hypot(self.ball.x - p.x, self.ball.y - p.y)
                if dist < PLAYER_RADIUS + BALL_RADIUS:
                    self.ball.possessed_by = p
                    self.ball.vx = 0.0
                    self.ball.vy = 0.0
                    break

    def _resolve_player_collisions(self) -> None:
        players = self._all_players
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
        for p in self._all_players:
            p.x = max(PITCH_L + PLAYER_RADIUS, min(PITCH_R - PLAYER_RADIUS, p.x))
            p.y = max(PITCH_B + PLAYER_RADIUS, min(PITCH_T - PLAYER_RADIUS, p.y))

    def _check_goals(self) -> None:
        goal_lo = PITCH_CY - GOAL_H / 2
        goal_hi = PITCH_CY + GOAL_H / 2

        if self.ball.x + BALL_RADIUS < PITCH_L and goal_lo <= self.ball.y <= goal_hi:
            self.score_b += 1
            self._trigger_goal_reset()
        elif self.ball.x - BALL_RADIUS > PITCH_R and goal_lo <= self.ball.y <= goal_hi:
            self.score_a += 1
            self._trigger_goal_reset()

    def _trigger_goal_reset(self) -> None:
        self._goal_flash = 1.5
        self.ball.reset()
        for p in self._all_players:
            p.reset()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    from miniball.ai import BaselineAI

    game = FootballGame(
        team_a_config=TeamConfig(
            name="Baseline model", ai=BaselineAI, human_controlled=None
        ),
        team_b_config=TeamConfig(name="Baseline model 2", ai=BaselineAI),
    )
    game.run()


if __name__ == "__main__":
    main()
