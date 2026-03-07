"""Miniball – 5-a-side football built with the arcade library.

Top-down view, no sprites: players are coloured circles, the ball is a white
circle.  One player (yellow ring) is controlled with the arrow keys.
"""

from __future__ import annotations

import math

import arcade

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

# ── Physics ──────────────────────────────────────────────────────────────────
PLAYER_RADIUS = 18
BALL_RADIUS = 10
PLAYER_SPEED = 220  # px / s
BALL_DRAG = 1.8  # speed loss per second (linear model: v -= drag * v * dt)
KICK_IMPULSE = 380  # px / s transferred to ball on contact
MAX_BALL_SPEED = 700

# ── Colours ──────────────────────────────────────────────────────────────────
C_GRASS = (34, 139, 34)
C_LINE = (255, 255, 255)
C_GOAL = (220, 220, 220)
C_BALL = (255, 255, 255)
C_BALL_OUTLINE = (20, 20, 20)
C_PLAYER_OUTLINE = (20, 20, 20)
C_CONTROLLED_RING = (255, 215, 0)
C_TEAM_A = (210, 40, 40)  # red  – left side
C_TEAM_B = (30, 100, 200)  # blue – right side
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

    def reset(self) -> None:
        self.x = PITCH_CX
        self.y = PITCH_CY
        self.vx = 0.0
        self.vy = 0.0

    def apply_impulse(self, ix: float, iy: float) -> None:
        self.vx += ix
        self.vy += iy
        speed = math.hypot(self.vx, self.vy)
        if speed > MAX_BALL_SPEED:
            scale = MAX_BALL_SPEED / speed
            self.vx *= scale
            self.vy *= scale

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Drag (time-based so it's frame-rate independent)
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

        # Left / right walls – bounce unless ball is in the goal opening
        if self.x - BALL_RADIUS < PITCH_L and not (goal_lo <= self.y <= goal_hi):
            self.x = PITCH_L + BALL_RADIUS
            self.vx = abs(self.vx)
        if self.x + BALL_RADIUS > PITCH_R and not (goal_lo <= self.y <= goal_hi):
            self.x = PITCH_R - BALL_RADIUS
            self.vx = -abs(self.vx)

        # Back wall of each goal box (ball can't exit through the net back)
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
        x: float,
        y: float,
        color: tuple[int, int, int],
        number: int,
        team: int,
    ) -> None:
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.color = color
        self.number = number
        self.team = team

    def reset(self) -> None:
        self.x = self.start_x
        self.y = self.start_y

    def draw(self, highlight: bool = False) -> None:
        arcade.draw_circle_filled(self.x, self.y, PLAYER_RADIUS, self.color)
        arcade.draw_circle_outline(self.x, self.y, PLAYER_RADIUS, C_PLAYER_OUTLINE, 2)
        if highlight:
            arcade.draw_circle_outline(
                self.x, self.y, PLAYER_RADIUS + 5, C_CONTROLLED_RING, 2
            )
        arcade.draw_text(
            str(self.number),
            self.x,
            self.y,
            C_LINE,
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
        self._goal_flash = 0.0  # seconds remaining to show goal flash

        for i, (x, y) in enumerate(TEAM_A_STARTS):
            self.team_a.append(Player(x, y, C_TEAM_A, i + 1, team=0))
        for i, (x, y) in enumerate(TEAM_B_STARTS):
            self.team_b.append(Player(x, y, C_TEAM_B, i + 1, team=1))

    @property
    def _controlled(self) -> Player:
        """The player controlled by keyboard input (team A forward #4)."""
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
            p.draw(highlight=(p is self._controlled))
        self._draw_hud()

    def _draw_pitch(self) -> None:
        lw = 2
        goal_lo = PITCH_CY - GOAL_H / 2
        goal_hi = PITCH_CY + GOAL_H / 2

        # Grass
        arcade.draw_lrbt_rectangle_filled(PITCH_L, PITCH_R, PITCH_B, PITCH_T, C_GRASS)

        # Pitch outline
        arcade.draw_lrbt_rectangle_outline(
            PITCH_L, PITCH_R, PITCH_B, PITCH_T, C_LINE, lw
        )

        # Centre line
        arcade.draw_line(PITCH_CX, PITCH_B, PITCH_CX, PITCH_T, C_LINE, lw)

        # Centre circle + spot
        arcade.draw_circle_outline(PITCH_CX, PITCH_CY, 70, C_LINE, lw)
        arcade.draw_circle_filled(PITCH_CX, PITCH_CY, 4, C_LINE)

        # Penalty areas
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

        # Goals (box behind goal line)
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
            "Arrow keys to move · You are Red #4 (yellow ring)",
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

    def on_key_release(self, key: int, modifiers: int) -> None:
        self._keys.discard(key)

    # ── Update ────────────────────────────────────────────────────────────────

    def on_update(self, delta_time: float) -> None:
        dt = min(delta_time, 1 / 30)

        if self._goal_flash > 0:
            self._goal_flash -= dt
            return  # freeze play during flash

        # Move controlled player
        dx = dy = 0.0
        if arcade.key.LEFT in self._keys:
            dx -= 1
        if arcade.key.RIGHT in self._keys:
            dx += 1
        if arcade.key.DOWN in self._keys:
            dy -= 1
        if arcade.key.UP in self._keys:
            dy += 1

        if dx != 0 or dy != 0:
            norm = math.hypot(dx, dy)
            self._controlled.x += (dx / norm) * PLAYER_SPEED * dt
            self._controlled.y += (dy / norm) * PLAYER_SPEED * dt

        # Clamp all players inside pitch
        for p in self._all_players:
            p.x = max(PITCH_L + PLAYER_RADIUS, min(PITCH_R - PLAYER_RADIUS, p.x))
            p.y = max(PITCH_B + PLAYER_RADIUS, min(PITCH_T - PLAYER_RADIUS, p.y))

        # Player–player collisions: positional separation for all pairs
        self._resolve_player_collisions()

        # Re-clamp after separation (a collision near a wall can push out of bounds)
        for p in self._all_players:
            p.x = max(PITCH_L + PLAYER_RADIUS, min(PITCH_R - PLAYER_RADIUS, p.x))
            p.y = max(PITCH_B + PLAYER_RADIUS, min(PITCH_T - PLAYER_RADIUS, p.y))

        # Ball–player collisions: push ball away from any overlapping player
        for p in self._all_players:
            contact_dist = PLAYER_RADIUS + BALL_RADIUS
            dist = math.hypot(self.ball.x - p.x, self.ball.y - p.y)
            if 0 < dist < contact_dist:
                angle = math.atan2(self.ball.y - p.y, self.ball.x - p.x)
                # Separate
                overlap = contact_dist - dist
                self.ball.x += math.cos(angle) * overlap
                self.ball.y += math.sin(angle) * overlap
                # Impulse is stronger for the player-controlled circle so it
                # feels responsive; other players give a gentler nudge.
                power = KICK_IMPULSE if p is self._controlled else KICK_IMPULSE * 0.35
                self.ball.apply_impulse(
                    math.cos(angle) * power,
                    math.sin(angle) * power,
                )

        self.ball.update(dt)
        self._check_goals()

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

    def _check_goals(self) -> None:
        goal_lo = PITCH_CY - GOAL_H / 2
        goal_hi = PITCH_CY + GOAL_H / 2

        # Goal when the ball has fully crossed the goal line (not the back of the net)
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
