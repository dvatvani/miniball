"""Live match view: renders the game and processes human input."""

from __future__ import annotations

import math
from typing import Literal

import arcade

from miniball.config import (
    BALL_DECEL,
    MAX_STRIKE_SPEED,
    PLAYER_RADIUS,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    STRIKE_ANGULAR_ERROR_DEGREES_FN,
    STRIKE_COOLDOWN,
    STRIKE_WEIGHT_CIRCLE,
    STRIKE_WEIGHT_CROSS,
    STRIKE_WEIGHT_SQUARE,
    STRIKE_WEIGHT_TRIANGLE,
    TACKLE_COOLDOWN,
)
from miniball.simulation.engine import HumanInput, MatchSimulation, Player
from miniball.teams import Team
from miniball.ui.config import (
    C_BALL,
    C_BALL_OUTLINE,
    C_CONTROLLED,
    C_COOLDOWN_RING,
    C_HINT,
    C_HUD,
    C_LINE,
    C_PLAYER_OUTLINE,
    COOLDOWN_ALPHA,
    JOY_SWITCH_THRESHOLD,
    PITCH_B,
    PITCH_L,
    PITCH_R,
    PITCH_T,
    SCREEN_BALL_RADIUS,
    SCREEN_H,
    SCREEN_PLAYER_RADIUS,
    SCREEN_W,
)
from miniball.ui.coords import global_to_screen
from miniball.ui.input import get_move_vector, get_strike_weight
from miniball.ui.pitch import draw_pitch

# Main-pitch pixel dimensions (used by draw_pitch)
_PITCH_PX_W = PITCH_R - PITCH_L
_PITCH_PX_H = PITCH_T - PITCH_B


class MatchView(arcade.View):
    """Active match view.  Hosts the simulation and renders the game."""

    def __init__(
        self,
        team_a_config: Team,
        team_b_config: Team,
        human_team: Literal["home", "away"] | None = None,
    ) -> None:
        super().__init__()
        self.human_team_side = human_team

        self.sim = MatchSimulation(team_a_config, team_b_config)

        if human_team == "home":
            self._human_team: list[Player] | None = self.sim.team_a
        elif human_team == "away":
            self._human_team = self.sim.team_b
        else:
            self._human_team = None
        self._controlled_idx: int | None = 0 if self._human_team is not None else None

        self._keys: set[int] = set()
        self._joy_switch_prev = False

        self._joystick = None
        self._joy_axis_state: dict[str, float] = {}

        self._stats_countdown: float = 1.0

    def on_show_view(self) -> None:
        arcade.set_background_color((30, 30, 30, 255))
        self._joystick = self.window.joystick
        self._joy_axis_state = self.window.joy_axis_state

    @property
    def _controlled(self) -> Player | None:
        if self._human_team is None or self._controlled_idx is None:
            return None
        return self._human_team[self._controlled_idx]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        draw_pitch(PITCH_L, PITCH_B, _PITCH_PX_W, _PITCH_PX_H, line_width=3)
        if self._controlled is not None:
            self._draw_aim_arc(self._controlled)
        self._draw_ball()
        for p in self.sim.all_players:
            self._draw_player(p, highlight=(p is self._controlled))
        self._draw_hud()

    def _draw_ball(self) -> None:
        sx, sy = global_to_screen(self.sim.ball.x, self.sim.ball.y)
        arcade.draw_circle_filled(sx, sy, SCREEN_BALL_RADIUS, C_BALL)
        arcade.draw_circle_outline(sx, sy, SCREEN_BALL_RADIUS, C_BALL_OUTLINE, 2)

    def _draw_player(self, p: Player, highlight: bool = False) -> None:
        sx, sy = global_to_screen(p.x, p.y)

        if p.on_cooldown:
            r, g, b = p.color[:3]
            fill_color: tuple[int, ...] = (r, g, b, COOLDOWN_ALPHA)
            outline_color: tuple[int, ...] = (*C_PLAYER_OUTLINE, COOLDOWN_ALPHA)
            text_color: tuple[int, ...] = (255, 255, 255, COOLDOWN_ALPHA)
        else:
            fill_color = p.color
            outline_color = C_PLAYER_OUTLINE
            text_color = C_LINE

        arcade.draw_circle_filled(sx, sy, SCREEN_PLAYER_RADIUS, fill_color)
        arcade.draw_circle_outline(sx, sy, SCREEN_PLAYER_RADIUS, outline_color, 2)

        if highlight:
            arcade.draw_circle_outline(
                sx, sy, SCREEN_PLAYER_RADIUS + 2, C_CONTROLLED, 2
            )

        if p.on_cooldown:
            _max_cd = max(TACKLE_COOLDOWN, STRIKE_COOLDOWN)
            fraction = min(1.0, p.cooldown_timer / _max_cd)
            _ring_r = SCREEN_PLAYER_RADIUS - 4
            if fraction >= 1.0:
                arcade.draw_circle_outline(sx, sy, _ring_r, C_COOLDOWN_RING, 3)
            elif fraction > 0.0:
                arcade.draw_arc_outline(
                    sx,
                    sy,
                    _ring_r * 2,
                    _ring_r * 2,
                    C_COOLDOWN_RING,
                    start_angle=90,
                    end_angle=90 + fraction * 360,
                    border_width=3,
                )

        arcade.draw_text(
            str(p.number),
            sx,
            sy,
            text_color,
            font_size=11,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

    def _draw_aim_arc(self, p: Player) -> None:
        gx = p.x + math.cos(p.facing) * PLAYER_RADIUS
        gy = p.y + math.sin(p.facing) * PLAYER_RADIUS

        def ray_t(angle: float, max_range: float) -> float:
            """Distance to pitch boundary or max_range along ``angle`` from (gx, gy)."""
            dx = math.cos(angle)
            dy = math.sin(angle)
            candidates: list[float] = []
            if abs(dx) > 1e-9:
                candidates.append(((STANDARD_PITCH_WIDTH if dx > 0 else 0) - gx) / dx)
            if abs(dy) > 1e-9:
                candidates.append(((STANDARD_PITCH_HEIGHT if dy > 0 else 0) - gy) / dy)
            valid = [tc for tc in candidates if tc > 1e-9]
            boundary_t = min(valid) if valid else 0.0
            return min(boundary_t, max_range)

        def draw_single_arc(
            weight: float,
            fill_color: tuple[int, int, int, int],
            line_color: tuple[int, int, int, int],
        ) -> None:
            error_rad = math.radians(STRIKE_ANGULAR_ERROR_DEGREES_FN(weight))
            # weight is a linear range fraction: range = weight * max_range
            max_range = weight * MAX_STRIKE_SPEED**2 / (2 * BALL_DECEL)
            angle_lo = p.facing - error_rad
            angle_hi = p.facing + error_rad

            n_steps = 16
            start_sx, start_sy = global_to_screen(gx, gy)
            points: list[tuple[float, float]] = [(start_sx, start_sy)]
            for i in range(n_steps + 1):
                angle = angle_lo + (angle_hi - angle_lo) * i / n_steps
                t = ray_t(angle, max_range)
                sx, sy = global_to_screen(
                    gx + math.cos(angle) * t, gy + math.sin(angle) * t
                )
                points.append((sx, sy))
            arcade.draw_polygon_filled(points, fill_color)

            t_centre = ray_t(p.facing, max_range)
            end_sx, end_sy = global_to_screen(
                gx + math.cos(p.facing) * t_centre,
                gy + math.sin(p.facing) * t_centre,
            )
            arcade.draw_line(start_sx, start_sy, end_sx, end_sy, line_color, 2)

        # Draw heaviest first so lighter arcs render on top
        # Square (full power) — PS4 pink
        draw_single_arc(STRIKE_WEIGHT_SQUARE, (220, 50, 150, 20), (220, 50, 150, 90))
        # Cross (balanced) — PS4 blue
        draw_single_arc(STRIKE_WEIGHT_CROSS, (50, 100, 210, 20), (50, 100, 210, 100))
        # Circle (light) — PS4 red
        draw_single_arc(STRIKE_WEIGHT_CIRCLE, (220, 40, 40, 20), (220, 40, 40, 90))
        # Triangle (tap) — PS4 green
        draw_single_arc(STRIKE_WEIGHT_TRIANGLE, (50, 190, 90, 20), (50, 190, 90, 90))

    def _draw_hud(self) -> None:
        secs = max(0.0, self.sim.time_remaining)
        mins = int(secs) // 60
        sec_part = int(secs) % 60
        timer_str = f"{mins}:{sec_part:02d}"
        arcade.draw_text(
            f"{self.sim.team_a_config.name}  {self.sim.score_a} – {self.sim.score_b}  {self.sim.team_b_config.name}",
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
            "Move: Arrows/L-stick  ·  \u25a1/Q power  \u2715/Spc balanced  \u25cb/E light  \u25b3/W tap  ·  Switch: Tab / R-stick",
            SCREEN_W / 2,
            28,
            C_HINT,
            font_size=12,
            anchor_x="center",
            anchor_y="center",
        )
        if self.sim.game_over:
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
        elif self.sim.goal_flash > 0:
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
        elif self.sim.countdown > 0:
            arcade.draw_text(
                str(math.ceil(self.sim.countdown)),
                SCREEN_W / 2,
                SCREEN_H / 2,
                (255, 220, 0),
                font_size=120,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

    # ── Input ─────────────────────────────────────────────────────────────────

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        self._keys.add(symbol)
        if symbol == arcade.key.TAB and self._human_team is not None:
            n = len(self._human_team)
            if n > 1 and self._controlled_idx is not None:
                step = -1 if (modifiers & arcade.key.MOD_SHIFT) else 1
                self._controlled_idx = (self._controlled_idx + step) % n
        return None

    def on_key_release(self, symbol: int, modifiers: int) -> bool | None:
        self._keys.discard(symbol)
        return None

    # ── Update ────────────────────────────────────────────────────────────────

    def on_update(self, delta_time: float) -> bool | None:
        dt = min(delta_time, 1 / 30)

        if self.sim.game_over:
            self._stats_countdown -= dt
            if self._stats_countdown <= 0:
                self._transition_to_stats()
            return

        self._handle_player_switch()

        in_pause = self.sim.goal_flash > 0 or self.sim.countdown > 0
        human_input: HumanInput | None = None
        if not in_pause:
            human_input = self._gather_human_input()

        self.sim.step(dt, human_input)

    def _transition_to_stats(self) -> None:
        from miniball.ui.stats_view import StatsView

        team_summary_df = None
        avg_positions_df = None
        saved_path = self.sim.export_history()
        if saved_path is not None:
            try:
                from miniball.db import create_db

                con = create_db()
                filename = str(saved_path)
                team_summary_df = con.execute(
                    "SELECT * FROM team_match WHERE filename = ?", [filename]
                ).pl()
                avg_positions_df = con.execute(
                    "SELECT player_number, is_home, avg_x, avg_y FROM player_match WHERE filename = ?",
                    [filename],
                ).pl()
                con.close()
            except Exception:
                pass

        self.window.show_view(
            StatsView(
                team_a=self.sim.team_a_config,
                team_b=self.sim.team_b_config,
                score_a=self.sim.score_a,
                score_b=self.sim.score_b,
                human_team_side=self.human_team_side,
                team_summary_df=team_summary_df,
                avg_positions_df=avg_positions_df,
            )
        )

    # ── Human input helpers ───────────────────────────────────────────────────

    def _handle_player_switch(self) -> None:
        if self._human_team is None or self._controlled is None:
            return
        if self._joystick is None:
            return
        jrx, jry = self._get_right_stick()
        switch_now = math.hypot(jrx, jry) > JOY_SWITCH_THRESHOLD
        if switch_now and not self._joy_switch_prev:
            self._switch_controlled_player(jrx, jry)
        self._joy_switch_prev = switch_now

    def _gather_human_input(self) -> HumanInput | None:
        if self._human_team is None or self._controlled_idx is None:
            return None
        controlled = self._controlled
        if controlled is None:
            return None

        dx, dy = get_move_vector(self._keys, self._joystick)
        is_home = self._human_team is self.sim.team_a
        weight = get_strike_weight(self._keys, self._joystick)

        return HumanInput(
            is_home=is_home,
            player_number=controlled.number,
            direction=(dx, dy),
            strike=weight is not None,
            strike_weight=weight if weight is not None else 0.5,
        )

    def _get_right_stick(self) -> tuple[float, float]:
        if self._joystick is None:
            return 0.0, 0.0
        state = self._joy_axis_state
        for x_attr, y_attr in (("z", "rz"), ("rx", "ry")):
            if x_attr in state or y_attr in state:
                return state.get(x_attr, 0.0), -state.get(y_attr, 0.0)
        return 0.0, 0.0

    def _switch_controlled_player(self, dx: float, dy: float) -> None:
        if self._human_team is None or self._controlled_idx is None:
            return
        current = self._controlled
        if current is None:
            return

        flick_angle = math.atan2(dy, dx)
        cx, cy = global_to_screen(current.x, current.y)
        best_idx: int | None = None
        best_diff = float("inf")

        for i, p in enumerate(self._human_team):
            if p is current:
                continue
            px, py = global_to_screen(p.x, p.y)
            angle_to = math.atan2(py - cy, px - cx)
            diff = abs(
                math.atan2(
                    math.sin(angle_to - flick_angle), math.cos(angle_to - flick_angle)
                )
            )
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx is not None:
            self._controlled_idx = best_idx
