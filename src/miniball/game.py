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
from typing import Literal

import arcade
import polars as pl

from miniball import match_stats
from miniball.config import (
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
    GOAL_DEPTH,
    GOAL_H,
    JOY_DEAD_ZONE,
    JOY_SWITCH_THRESHOLD,
    PITCH_B,
    PITCH_CX,
    PITCH_CY,
    PITCH_L,
    PITCH_R,
    PITCH_T,
    PLAYER_RADIUS,
    SCREEN_H,
    SCREEN_W,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    TITLE,
)
from miniball.simulation import GameSimulation, HumanInput, Player
from miniball.teams import Team

# ── Main window ───────────────────────────────────────────────────────────────


class FootballGame(arcade.Window):
    def __init__(
        self,
        team_a_config: Team,
        team_b_config: Team,
        human_team: Literal["home", "away"] | None = None,
    ) -> None:
        super().__init__(SCREEN_W, SCREEN_H, TITLE)
        arcade.set_background_color((30, 30, 30, 255))

        # Core simulation (all physics, AI, history recording)
        self.sim = GameSimulation(team_a_config, team_b_config)

        # Which team list the human player belongs to (None = fully AI game).
        # Index within that list of the currently controlled player (starts on GK = 0).
        if human_team == "home":
            self._human_team: list[Player] | None = self.sim.team_a
        elif human_team == "away":
            self._human_team = self.sim.team_b
        else:
            self._human_team = None
        self._controlled_idx: int | None = 0 if self._human_team is not None else None

        # Input state
        self._keys: set[int] = set()
        self._joy_shoot_prev = False
        self._joy_switch_prev = False
        self._human_shoot_requested = (
            False  # set for one frame when human presses shoot
        )

        # Gamepad – use the first connected controller if one is present
        joysticks = arcade.get_joysticks()  # type: ignore[attr-defined]
        self._joystick = joysticks[0] if joysticks else None
        # Live dict of axis values populated by on_joyaxis_motion events.
        # Used by _get_right_stick to auto-detect which axis pair is the right stick.
        self._joy_axis_state: dict[str, float] = {}
        if self._joystick is not None:
            self._joystick.open()
            game_self = self  # closure reference

            @self._joystick.event
            def on_joyaxis_motion(joystick, axis: str, value: float) -> None:  # noqa: ANN001
                """Keep a live snapshot of every axis."""
                game_self._joy_axis_state[axis] = value

        # Post-game stats display
        self._stats_countdown: float = (
            3.0  # delay after final whistle before showing stats
        )
        self._show_stats: bool = False
        self._team_summary_df: pl.DataFrame | None = None
        self._avg_positions_df: pl.DataFrame | None = None

    @property
    def _controlled(self) -> Player | None:
        """Human-controlled player, or ``None`` when the game is fully AI-driven."""
        if self._human_team is None or self._controlled_idx is None:
            return None
        return self._human_team[self._controlled_idx]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        if self._show_stats:
            self._draw_stats_screen()
            return
        self._draw_pitch()
        self._draw_ball()
        for p in self.sim.all_players:
            self._draw_player(
                p,
                highlight=(p is self._controlled),
                has_ball=(self.sim.ball.possessed_by is p),
            )
        self._draw_hud()

    def _draw_ball(self) -> None:
        arcade.draw_circle_filled(self.sim.ball.x, self.sim.ball.y, BALL_RADIUS, C_BALL)
        arcade.draw_circle_outline(
            self.sim.ball.x, self.sim.ball.y, BALL_RADIUS, C_BALL_OUTLINE, 2
        )

    def _draw_player(
        self, p: Player, highlight: bool = False, has_ball: bool = False
    ) -> None:
        if p.on_cooldown:
            r, g, b = p.color[:3]
            fill_color: tuple[int, ...] = (r, g, b, COOLDOWN_ALPHA)
            outline_color: tuple[int, ...] = (*C_PLAYER_OUTLINE, COOLDOWN_ALPHA)
            text_color: tuple[int, ...] = (255, 255, 255, COOLDOWN_ALPHA)
        else:
            fill_color = p.color
            outline_color = C_PLAYER_OUTLINE
            text_color = C_LINE

        if has_ball:
            arcade.draw_circle_outline(p.x, p.y, PLAYER_RADIUS + 5, C_POSSESSION, 3)

        arcade.draw_circle_filled(p.x, p.y, PLAYER_RADIUS, fill_color)
        arcade.draw_circle_outline(p.x, p.y, PLAYER_RADIUS, outline_color, 2)

        if highlight:
            arcade.draw_circle_outline(p.x, p.y, PLAYER_RADIUS + 2, C_CONTROLLED, 2)

        arcade.draw_text(
            str(p.number),
            p.x,
            p.y,
            text_color,
            font_size=11,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

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
            "Arrows / L-stick to move and aim shot · Space / A to shoot · R-stick to switch controlled player",
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

    def _draw_stats_screen(self) -> None:
        """Render the post-match summary statistics screen.

        Layout
        ──────
        Left  ~45 %  Mini pitch with average-position dots + colour legend.
        Right ~55 %  Team names header + one bar-chart row per statistic,
                     with each team's value displayed on its own side of the bar.
        """
        # ── Background ───────────────────────────────────────────────────────
        arcade.draw_lrbt_rectangle_filled(0, SCREEN_W, 0, SCREEN_H, (15, 35, 15))

        # ── Header (full-width) ───────────────────────────────────────────────
        arcade.draw_text(
            "FULL TIME",
            SCREEN_W / 2,
            763,
            (255, 220, 0),
            font_size=52,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        arcade.draw_text(
            f"{self.sim.team_a_config.name}  {self.sim.score_a} – {self.sim.score_b}"
            f"  {self.sim.team_b_config.name}",
            SCREEN_W / 2,
            715,
            (255, 255, 255),
            font_size=24,
            anchor_x="center",
            anchor_y="center",
        )

        # ── Mini pitch (left, 480 × 320, scale = 4 px/unit) ──────────────────
        _S = 4
        _PW = STANDARD_PITCH_WIDTH * _S  # 480
        _PH = STANDARD_PITCH_HEIGHT * _S  # 320
        _PL = 30
        _PB = 200
        _PT = _PB + _PH  # 520
        _PCX = _PL + _PW // 2  # 270
        _PCY = _PB + _PH // 2  # 360

        _mpw = PITCH_R - PITCH_L  # main-pitch pixel width
        _mph = PITCH_T - PITCH_B  # main-pitch pixel height
        _goal_h = round(GOAL_H / _mph * STANDARD_PITCH_HEIGHT * _S)
        _goal_d = round(GOAL_DEPTH / _mpw * STANDARD_PITCH_WIDTH * _S)
        _circle_r = round(70 / _mpw * STANDARD_PITCH_WIDTH * _S)

        arcade.draw_lrbt_rectangle_filled(_PL, _PL + _PW, _PB, _PT, C_GRASS)
        arcade.draw_lrbt_rectangle_outline(_PL, _PL + _PW, _PB, _PT, C_LINE, 2)
        arcade.draw_line(_PCX, _PB, _PCX, _PT, C_LINE, 1)
        arcade.draw_circle_outline(_PCX, _PCY, _circle_r, C_LINE, 1)

        _g_lo = _PCY - _goal_h // 2
        _g_hi = _PCY + _goal_h // 2
        # Away goal (left – global x = 0)
        arcade.draw_lrbt_rectangle_filled(_PL - _goal_d, _PL, _g_lo, _g_hi, C_GOAL)
        arcade.draw_lrbt_rectangle_outline(_PL - _goal_d, _PL, _g_lo, _g_hi, C_LINE, 1)
        # Home goal (right – global x = 120)
        arcade.draw_lrbt_rectangle_filled(
            _PL + _PW, _PL + _PW + _goal_d, _g_lo, _g_hi, C_GOAL
        )
        arcade.draw_lrbt_rectangle_outline(
            _PL + _PW, _PL + _PW + _goal_d, _g_lo, _g_hi, C_LINE, 1
        )

        # Average-position dots
        if self._avg_positions_df is not None:
            for row in self._avg_positions_df.iter_rows(named=True):
                dot_x = _PL + row["avg_x"] * _S
                dot_y = _PB + row["avg_y"] * _S
                dot_color: tuple[int, int, int] = (
                    C_TEAM_A if row["is_home"] else C_TEAM_B
                )
                arcade.draw_circle_filled(dot_x, dot_y, 7, dot_color)
                arcade.draw_circle_outline(dot_x, dot_y, 7, (20, 20, 20), 1)

        # Colour legend below pitch
        _leg_y = _PB - 32
        arcade.draw_circle_filled(_PL + 8, _leg_y + 5, 6, C_TEAM_A)
        arcade.draw_text(
            f"  {self.sim.team_a_config.name} (home)",
            _PL + 14,
            _leg_y - 3,
            C_TEAM_A,
            13,
        )
        arcade.draw_circle_filled(_PL + _PW // 2 + 8, _leg_y + 5, 6, C_TEAM_B)
        arcade.draw_text(
            f"  {self.sim.team_b_config.name} (away)",
            _PL + _PW // 2 + 14,
            _leg_y - 3,
            C_TEAM_B,
            13,
        )

        # ── Stats panel (right) ───────────────────────────────────────────────
        _SX = 545  # left edge of stats panel
        _SR = 1175  # right edge

        if self._team_summary_df is None:
            return
        try:
            home = self._team_summary_df.filter(pl.col("is_home")).row(0, named=True)
            away = self._team_summary_df.filter(~pl.col("is_home")).row(0, named=True)
        except Exception:
            return

        # Vertically align the stats table with the pitch on the left.
        # _PT / _PB are the pitch top / bottom pixel coordinates computed above.
        _hdr_y = _PT - 14  # team-name header centre
        _sep_y = _PT - 30  # divider line
        _top_row_y = _PT - 62  # centre of first stat row
        _bot_row_y = _PB + 20  # centre of last  stat row

        # Team-name header row
        arcade.draw_text(
            str(home["team"]), _SX, _hdr_y, C_TEAM_A, 18, bold=True, anchor_y="center"
        )
        arcade.draw_text(
            str(away["team"]),
            _SR,
            _hdr_y,
            C_TEAM_B,
            18,
            bold=True,
            anchor_x="right",
            anchor_y="center",
        )
        arcade.draw_line(_SX, _sep_y, _SR, _sep_y, (80, 110, 80), 1)

        # One bar-chart row per statistic
        def _fval(v: int | float | None) -> float:
            """Coerce a possibly-null stat value to float."""
            return float(v) if v is not None else 0.0

        stat_rows: list[tuple[str, float, float, str]] = [
            ("Goals", _fval(home["goals"]), _fval(away["goals"]), "{:.0f}"),
            ("Shots", _fval(home["shots"]), _fval(away["shots"]), "{:.0f}"),
            (
                "Possession",
                _fval(home["possession_pct"]),
                _fval(away["possession_pct"]),
                "{:.1f}%",
            ),
            (
                "Avg poss. duration",
                _fval(home["avg_duration"]),
                _fval(away["avg_duration"]),
                "{:.1f}s",
            ),
        ]
        # Distribute rows evenly between _top_row_y and _bot_row_y.
        n = len(stat_rows)
        dynamic_step = (_top_row_y - _bot_row_y) / (n - 1) if n > 1 else 0.0
        for i, (label, h_val, a_val, fmt) in enumerate(stat_rows):
            self._draw_stat_bar_row(
                _SX, _SR, _top_row_y - i * dynamic_step, label, h_val, a_val, fmt
            )

    def _draw_stat_bar_row(
        self,
        panel_x: float,
        panel_r: float,
        row_y: float,
        label: str,
        home_val: float,
        away_val: float,
        fmt: str,
    ) -> None:
        """Draw one statistic row: label | home value | stacked bar | away value.

        The stacked bar is split left/right in proportion to the two teams' values.
        A thin guide line marks the 50 % mid-point for quick visual reference.
        """
        label_w: float = 150
        val_w: float = 80
        gap: float = 10
        bar_left = panel_x + label_w + val_w + gap
        bar_right = panel_r - val_w - gap
        bar_w = bar_right - bar_left  # ~300 px
        bar_h: float = 30

        # Stat label
        arcade.draw_text(label, panel_x, row_y, C_HINT, 15, anchor_y="center")

        # Numeric values (coloured, bold, flanking the bar)
        arcade.draw_text(
            fmt.format(home_val),
            bar_left - gap,
            row_y,
            C_TEAM_A,
            16,
            anchor_x="right",
            anchor_y="center",
            bold=True,
        )
        arcade.draw_text(
            fmt.format(away_val),
            bar_right + gap,
            row_y,
            C_TEAM_B,
            16,
            anchor_y="center",
            bold=True,
        )

        # Stacked bar fill
        total = home_val + away_val
        home_frac = home_val / total if total > 0 else 0.5
        home_w = bar_w * home_frac
        bar_lo = row_y - bar_h / 2
        bar_hi = row_y + bar_h / 2

        home_fill: tuple[int, int, int, int] = (
            C_TEAM_A[0],
            C_TEAM_A[1],
            C_TEAM_A[2],
            200,
        )
        away_fill: tuple[int, int, int, int] = (
            C_TEAM_B[0],
            C_TEAM_B[1],
            C_TEAM_B[2],
            200,
        )

        if home_w > 0:
            arcade.draw_lrbt_rectangle_filled(
                bar_left, bar_left + home_w, bar_lo, bar_hi, home_fill
            )
        if home_w < bar_w:
            arcade.draw_lrbt_rectangle_filled(
                bar_left + home_w, bar_right, bar_lo, bar_hi, away_fill
            )

        # Border + 50 % guide
        arcade.draw_lrbt_rectangle_outline(
            bar_left, bar_right, bar_lo, bar_hi, C_HINT, 1
        )
        mid_x = bar_left + bar_w / 2
        arcade.draw_line(mid_x, bar_lo + 2, mid_x, bar_hi - 2, (160, 160, 160), 1)

    # ── Input ─────────────────────────────────────────────────────────────────

    def on_key_press(self, key: int, modifiers: int) -> None:
        self._keys.add(key)
        if key == arcade.key.SPACE:
            self._human_shoot_requested = True

    def on_key_release(self, key: int, modifiers: int) -> None:
        self._keys.discard(key)

    # ── Update ────────────────────────────────────────────────────────────────

    def on_update(self, delta_time: float) -> None:
        dt = min(delta_time, 1 / 30)

        if self.sim.game_over:
            if not self._show_stats:
                self._stats_countdown -= dt
                if self._stats_countdown <= 0:
                    match_df = self.sim.build_match_df()
                    if match_df is not None:
                        self.sim._write_parquet(match_df)
                        self._team_summary_df = match_stats.team_summary(match_df)
                        self._avg_positions_df = match_stats.avg_positions(match_df)
                    self._show_stats = True
            return

        # Only process human input during active play (not during goal flash or countdown)
        in_pause = self.sim.goal_flash > 0 or self.sim.countdown > 0
        human_input: HumanInput | None = None
        if not in_pause:
            human_input = self._gather_human_input()

        # Advance simulation
        self.sim.step(dt, human_input)

        # Clear single-frame shoot flag now that it has been consumed
        self._human_shoot_requested = False

    # ── Human input helpers ───────────────────────────────────────────────────

    def _gather_human_input(self) -> HumanInput | None:
        """Collect the current frame's human input.

        Returns ``None`` when no human player is configured.  Handles
        gamepad shoot-button edge detection and right-stick player switching
        as side effects before packaging the result.
        """
        if self._human_team is None:
            return None

        # Gamepad: shoot button (edge-triggered) + right-stick player switch
        if self._controlled is not None and self._joystick is not None:
            if self._joystick.buttons:
                shoot_now = bool(self._joystick.buttons[0])
                if shoot_now and not self._joy_shoot_prev:
                    self._human_shoot_requested = True
                self._joy_shoot_prev = shoot_now

            jrx, jry = self._get_right_stick()
            switch_now = math.hypot(jrx, jry) > JOY_SWITCH_THRESHOLD
            if switch_now and not self._joy_switch_prev:
                self._switch_controlled_player(jrx, jry)
            self._joy_switch_prev = switch_now

        if self._controlled_idx is None:
            return None
        controlled = self._controlled
        if controlled is None:
            return None

        dx, dy = self._get_move_input()
        is_home = self._human_team is self.sim.team_a

        return HumanInput(
            is_home=is_home,
            player_number=controlled.number,
            direction=(dx, dy),
            shoot=self._human_shoot_requested,
        )

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

    def _get_right_stick(self) -> tuple[float, float]:
        """Return the (dx, dy) vector of the right analogue stick.

        Pyglet pre-initialises ALL axis attributes to 0 on every Joystick
        object regardless of whether the hardware has them, so ``hasattr``
        cannot distinguish real axes from phantom ones.  Instead we rely on
        ``_joy_axis_state``, which is populated only when the OS actually
        fires an ``on_joyaxis_motion`` event, making it a reliable proxy for
        "this axis physically exists on this device".

        Candidate pairs tried in order (first whose keys appear in the state
        dict wins):
          • ('z', 'rz')  – PS / generic gamepads on macOS
          • ('rx', 'ry') – Xbox / standard HID
        Applies the same Pyglet Y-axis inversion as the left stick.
        """
        if self._joystick is None:
            return 0.0, 0.0
        state = self._joy_axis_state
        for x_attr, y_attr in (("z", "rz"), ("rx", "ry")):
            if x_attr in state or y_attr in state:
                return state.get(x_attr, 0.0), -state.get(y_attr, 0.0)
        return 0.0, 0.0

    def _switch_controlled_player(self, dx: float, dy: float) -> None:
        """Switch human control to the teammate whose position is most in the direction (dx, dy).

        The teammate with the smallest angular difference between the flick
        direction and the vector from the current controlled player to that
        teammate is selected.
        """
        if self._human_team is None or self._controlled_idx is None:
            return
        current = self._controlled
        if current is None:
            return

        flick_angle = math.atan2(dy, dx)
        best_idx: int | None = None
        best_diff = float("inf")

        for i, p in enumerate(self._human_team):
            if p is current:
                continue
            angle_to = math.atan2(p.y - current.y, p.x - current.x)
            # Wrap angular difference to [-π, π]
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


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    import questionary

    from miniball.teams import teams, teams_list

    team_names = [t.name for t in teams_list]

    human_side = questionary.select(
        "Control a player on a team with a controller (recommended) or keyboard?",
        choices=["home", "away", "none (watch AI match)"],
    ).ask()

    if human_side in ("home", "away"):
        print(f"You will control one player on the {human_side} team")
        print(
            "The controlled player can be changed in-game, but an AI needs to be "
            "selected for your team's other players."
        )
    home_name = questionary.select(
        "Home team AI (Starts with ball):",
        choices=team_names,
    ).ask()

    away_name = questionary.select(
        "Away team AI:",
        choices=[n for n in team_names if n != home_name],
    ).ask()

    human_team: Literal["home", "away"] | None = None
    if human_side == "home":
        human_team = "home"
    elif human_side == "away":
        human_team = "away"

    game = FootballGame(
        team_a_config=teams[home_name],
        team_b_config=teams[away_name],
        human_team=human_team,
    )
    game.run()


if __name__ == "__main__":
    main()
