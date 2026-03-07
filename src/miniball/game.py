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

from miniball.ai import (
    BaseAI,
    BaselineAI,
    GameState,
    PitchInfo,
    PlayerState,
    TeamActions,
)

# ── Window ──────────────────────────────────────────────────────────────────
SCREEN_W = 1200
SCREEN_H = 800
TITLE = "Miniball – 5-a-side Football"

# ── Pitch geometry ───────────────────────────────────────────────────────────
PITCH_L = 100
PITCH_R = 1100
PITCH_B = 75
PITCH_T = 725
PITCH_CX = (PITCH_L + PITCH_R) / 2
PITCH_CY = (PITCH_B + PITCH_T) / 2
GOAL_H = 140  # vertical opening of each goal
GOAL_DEPTH = 32  # how far the goal box extends behind the goal line

# ── Physics / timings ────────────────────────────────────────────────────────
PLAYER_RADIUS = 18
BALL_RADIUS = 10
PLAYER_SPEED = 180  # px / s
BALL_DRAG = 1.3  # speed loss per second (free ball, linear model)
SHOOT_SPEED = 750  # px / s on a Space-bar kick
MAX_BALL_SPEED = 700
TACKLE_COOLDOWN = 1.0  # seconds unable to gain the ball after being tackled
SHOT_COOLDOWN = 0.4  # seconds before the kicker can re-absorb their own shot
COOLDOWN_ALPHA = 90  # draw opacity (0–255) while on cooldown (~35 %)
JOY_DEAD_ZONE = 0.15  # ignore analogue stick values below this magnitude

# ── Colours ──────────────────────────────────────────────────────────────────
C_GRASS = (34, 139, 34)
C_LINE = (255, 255, 255)
C_GOAL = (220, 220, 220)
C_BALL = (255, 255, 255)
C_BALL_OUTLINE = (20, 20, 20)
C_PLAYER_OUTLINE = (20, 20, 20)
C_CONTROLLED = (255, 215, 0)  # yellow  – keyboard-controlled player
C_POSSESSION = (255, 255, 255)  # white   – player who has the ball
C_TEAM_A = (210, 40, 40)  # red   – left side
C_TEAM_B = (30, 100, 200)  # blue  – right side
C_HUD = (255, 255, 255)
C_HINT = (180, 180, 180)

# ── Starting positions  (1 GK + 2 defenders + 2 forwards, mirrored) ─────────
TEAM_A_STARTS: list[tuple[float, float]] = [
    (PITCH_L + 60, PITCH_CY),
    (PITCH_L + 230, PITCH_CY - 120),
    (PITCH_L + 230, PITCH_CY + 120),
    (PITCH_CX - 180, PITCH_CY - 90),
    (PITCH_CX - 180, PITCH_CY + 90),
]
TEAM_B_STARTS: list[tuple[float, float]] = [
    (PITCH_R - 60, PITCH_CY),
    (PITCH_R - 230, PITCH_CY - 120),
    (PITCH_R - 230, PITCH_CY + 120),
    (PITCH_CX + 180, PITCH_CY - 90),
    (PITCH_CX + 180, PITCH_CY + 90),
]


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
        player_id: str,
        x: float,
        y: float,
        color: tuple[int, int, int],
        number: int,
        team: int,
    ) -> None:
        self.player_id = player_id
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.color = color
        self.number = number
        self.team = team
        # Face toward the opposing half at kick-off
        self.facing: float = 0.0 if team == 0 else math.pi
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
        self.facing = 0.0 if self.team == 0 else math.pi
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
    def __init__(self) -> None:
        super().__init__(SCREEN_W, SCREEN_H, TITLE)
        arcade.set_background_color((30, 30, 30, 255))

        self.ball = Ball()
        self.team_a: list[Player] = []
        self.team_b: list[Player] = []
        self.score_a = 0
        self.score_b = 0
        self._keys: set[int] = set()
        self._goal_flash = 0.0

        # Gamepad – use the first connected controller if one is present
        joysticks = arcade.get_joysticks()
        self._joystick = joysticks[0] if joysticks else None
        if self._joystick is not None:
            self._joystick.open()
        self._joy_shoot_prev = False  # edge-detect the shoot button

        for i, (x, y) in enumerate(TEAM_A_STARTS):
            self.team_a.append(Player(f"A{i + 1}", x, y, C_TEAM_A, i + 1, team=0))
        for i, (x, y) in enumerate(TEAM_B_STARTS):
            self.team_b.append(Player(f"B{i + 1}", x, y, C_TEAM_B, i + 1, team=1))

        # AI engines – swap these out to change team behaviour.
        # Team A's non-human players use StationaryAI; team B uses BaselineAI.
        self._ai_a: BaseAI = BaselineAI(team="A")
        self._ai_b: BaseAI = BaselineAI(team="B")

    @property
    def _controlled(self) -> Player:
        """Keyboard-controlled player – Red #4."""
        return self.team_a[3]

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
        arcade.draw_text(
            f"Red  {self.score_a} – {self.score_b}  Blue",
            SCREEN_W / 2,
            SCREEN_H - 32,
            C_HUD,
            font_size=22,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        arcade.draw_text(
            "Arrows / L-stick · Space / A to shoot · You are Red #4 (yellow ring)",
            SCREEN_W / 2,
            28,
            C_HINT,
            font_size=12,
            anchor_x="center",
            anchor_y="center",
        )
        if self._goal_flash > 0:
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

        if self._goal_flash > 0:
            self._goal_flash -= dt
            return

        # 1. Tick all player timers
        for p in self._all_players:
            p.tick(dt)

        # 2. Move the controlled player
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

        # 2b. AI move + shoot for non-human players
        #     Build state once per team (is_teammate flips between the two).
        state_a = self._build_game_state("A")
        ai_a_actions = self._ai_a.get_actions(state_a)
        self._apply_ai_actions(
            [p for p in self.team_a if p is not self._controlled],
            ai_a_actions,
            dt,
        )
        state_b = self._build_game_state("B")
        ai_b_actions = self._ai_b.get_actions(state_b)
        self._apply_ai_actions(self.team_b, ai_b_actions, dt)

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

    def _build_game_state(self, perspective_team: str) -> GameState:
        """Snapshot the current game into the AI state dict.

        ``is_teammate`` is set to ``True`` for players on ``perspective_team``
        so each AI only needs to check that flag rather than compare team IDs.
        """
        possessor_id = (
            self.ball.possessed_by.player_id
            if self.ball.possessed_by is not None
            else None
        )
        players: list[PlayerState] = [
            {
                "player_id": p.player_id,
                "team": "A" if p.team == 0 else "B",
                "is_teammate": (("A" if p.team == 0 else "B") == perspective_team),
                "has_ball": self.ball.possessed_by is p,
                "on_cooldown": p.on_cooldown,
                "location": [p.x, p.y],
                "facing": p.facing,
            }
            for p in self._all_players
        ]
        pitch: PitchInfo = {
            "left": PITCH_L,
            "right": PITCH_R,
            "bottom": PITCH_B,
            "top": PITCH_T,
            "goal_height": GOAL_H,
            "attacking_direction": 1 if perspective_team == "A" else -1,
        }
        return {
            "players": players,
            "ball": {
                "location": [self.ball.x, self.ball.y],
                "velocity": [self.ball.vx, self.ball.vy],
                "possessed_by": possessor_id,
            },
            "pitch": pitch,
        }

    def _apply_ai_actions(
        self,
        players: list[Player],
        actions: TeamActions,
        dt: float,
    ) -> None:
        """Move and optionally shoot for each player according to AI actions."""
        for p in players:
            action = actions.get(p.player_id)
            if action is None:
                continue
            dx, dy = action["move"]
            if dx != 0 or dy != 0:
                norm = math.hypot(dx, dy)
                # Allow sub-1 magnitude from analogue-style AI outputs
                speed_frac = min(norm, 1.0)
                p.x += (dx / norm) * PLAYER_SPEED * speed_frac * dt
                p.y += (dy / norm) * PLAYER_SPEED * speed_frac * dt
                p.facing = math.atan2(dy, dx)
            if action["shoot"]:
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
        Silently does nothing if the given player doesn't have the ball.
        """
        if player is None:
            player = self._controlled
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
                if p.team == possessor.team:
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
    game = FootballGame()
    game.run()


if __name__ == "__main__":
    main()
