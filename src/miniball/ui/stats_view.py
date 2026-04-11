"""Post-match statistics screen with action buttons."""

from __future__ import annotations

from typing import Literal

import arcade
import polars as pl

from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH
from miniball.coords import team_to_global
from miniball.teams import Team
from miniball.ui.config import C_CONTROLLED, C_HINT, C_HUD, C_TEAM_A, C_TEAM_B, SCREEN_W
from miniball.ui.input import Action, ControllerPoller, actions_from_key
from miniball.ui.pitch import draw_pitch

# ── Layout constants ──────────────────────────────────────────────────────────

_SCALE = 4
_PITCH_W = STANDARD_PITCH_WIDTH * _SCALE
_PITCH_H = STANDARD_PITCH_HEIGHT * _SCALE
_PITCH_L = 30
_PITCH_B = 200

BTN_CY = 100
BTN_HW = 105
BTN_HH = 25
BTN_CXS = (225, 600, 975)
BTN_LABELS = ("Replay", "Team Select", "Quit")


class StatsView(arcade.View):
    """Full-time statistics and action buttons."""

    def __init__(
        self,
        team_a: Team,
        team_b: Team,
        score_a: int,
        score_b: int,
        human_team_side: Literal["home", "away"] | None,
        team_summary_df: pl.DataFrame | None,
        avg_positions_df: pl.DataFrame | None,
    ) -> None:
        super().__init__()
        self._team_a = team_a
        self._team_b = team_b
        self._score_a = score_a
        self._score_b = score_b
        self._human_team_side = human_team_side
        self._team_summary_df = team_summary_df
        self._avg_positions_df = avg_positions_df

        self._focus = 0  # 0 = Replay, 1 = Team Select, 2 = Quit
        self._poller = ControllerPoller()

    def on_show_view(self) -> None:
        arcade.set_background_color((30, 30, 30, 255))

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        self._draw_header()
        self._draw_mini_pitch()
        self._draw_stats_panel()
        self._draw_buttons()

    def _draw_header(self) -> None:
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
            f"{self._team_a.name}  {self._score_a} – {self._score_b}"
            f"  {self._team_b.name}",
            SCREEN_W / 2,
            715,
            (255, 255, 255),
            font_size=24,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_mini_pitch(self) -> None:
        draw_pitch(_PITCH_L, _PITCH_B, _PITCH_W, _PITCH_H, line_width=2)

        if self._avg_positions_df is not None:
            for row in self._avg_positions_df.iter_rows(named=True):
                gx, gy = team_to_global(
                    row["avg_x"], row["avg_y"], is_home=row["is_home"]
                )
                dot_x = _PITCH_L + gx * _SCALE
                dot_y = _PITCH_B + gy * _SCALE
                dot_color: tuple[int, int, int] = (
                    C_TEAM_A if row["is_home"] else C_TEAM_B
                )
                arcade.draw_circle_filled(dot_x, dot_y, 7, dot_color)
                arcade.draw_circle_outline(dot_x, dot_y, 7, (20, 20, 20), 1)

        leg_y = _PITCH_B - 32
        arcade.draw_circle_filled(_PITCH_L + 8, leg_y + 5, 6, C_TEAM_A)
        arcade.draw_text(
            f"  {self._team_a.name} (home)",
            _PITCH_L + 14,
            leg_y - 3,
            C_TEAM_A,
            13,
        )
        arcade.draw_circle_filled(_PITCH_L + _PITCH_W // 2 + 8, leg_y + 5, 6, C_TEAM_B)
        arcade.draw_text(
            f"  {self._team_b.name} (away)",
            _PITCH_L + _PITCH_W // 2 + 14,
            leg_y - 3,
            C_TEAM_B,
            13,
        )

    def _draw_stats_panel(self) -> None:
        sx = 545
        sr = 1175
        pt = _PITCH_B + _PITCH_H

        if self._team_summary_df is None:
            return
        try:
            home = self._team_summary_df.filter(pl.col("is_home")).row(0, named=True)
            away = self._team_summary_df.filter(~pl.col("is_home")).row(0, named=True)
        except Exception:
            return

        hdr_y = pt - 14
        sep_y = pt - 30
        top_row_y = pt - 62
        bot_row_y = _PITCH_B + 20

        arcade.draw_text(
            str(home["team_name"]),
            sx,
            hdr_y,
            C_TEAM_A,
            18,
            bold=True,
            anchor_y="center",
        )
        arcade.draw_text(
            str(away["team_name"]),
            sr,
            hdr_y,
            C_TEAM_B,
            18,
            bold=True,
            anchor_x="right",
            anchor_y="center",
        )
        arcade.draw_line(sx, sep_y, sr, sep_y, (80, 110, 80), 1)

        def _fval(v: int | float | None) -> float:
            return float(v) if v is not None else 0.0

        stat_rows: list[tuple[str, float, float, str]] = [
            ("Goals", _fval(home["goals"]), _fval(away["goals"]), "{:.0f}"),
            (
                "Shots on target",
                _fval(home["shots_on_target"]),
                _fval(away["shots_on_target"]),
                "{:.0f}",
            ),
            ("Shots", _fval(home["shots"]), _fval(away["shots"]), "{:.0f}"),
            (
                "Shot conversion",
                _fval(home["shot_conversion_rate"]),
                _fval(away["shot_conversion_rate"]),
                "{:.1f}%",
            ),
            (
                "Passes completed",
                _fval(home["passes_completed"]),
                _fval(away["passes_completed"]),
                "{:.0f}",
            ),
            (
                "Pass accuracy",
                _fval(home["pass_accuracy"]),
                _fval(away["pass_accuracy"]),
                "{:.1f}%",
            ),
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
        n = len(stat_rows)
        step = (top_row_y - bot_row_y) / (n - 1) if n > 1 else 0.0
        for i, (label, h_val, a_val, fmt) in enumerate(stat_rows):
            _draw_stat_bar_row(sx, sr, top_row_y - i * step, label, h_val, a_val, fmt)

    def _draw_buttons(self) -> None:
        for i, (cx, label) in enumerate(zip(BTN_CXS, BTN_LABELS)):
            focused = i == self._focus
            fill = (55, 55, 80) if focused else (38, 38, 55)
            border = C_CONTROLLED if focused else C_HINT
            arcade.draw_lrbt_rectangle_filled(
                cx - BTN_HW, cx + BTN_HW, BTN_CY - BTN_HH, BTN_CY + BTN_HH, fill
            )
            arcade.draw_lrbt_rectangle_outline(
                cx - BTN_HW, cx + BTN_HW, BTN_CY - BTN_HH, BTN_CY + BTN_HH, border, 2
            )
            arcade.draw_text(
                label,
                cx,
                BTN_CY,
                C_HUD if focused else C_HINT,
                16,
                anchor_x="center",
                anchor_y="center",
            )

    # ── Input ─────────────────────────────────────────────────────────────────

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self._replay()
            return None
        if symbol == arcade.key.T:
            self._go_to_team_select()
            return None
        if symbol in (arcade.key.Q, arcade.key.ESCAPE):
            arcade.exit()
            return None
        for action in actions_from_key(symbol, modifiers):
            self._handle_action(action)
        return None

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        for i, cx in enumerate(BTN_CXS):
            if abs(x - cx) <= BTN_HW and abs(y - BTN_CY) <= BTN_HH:
                if i == 0:
                    self._replay()
                elif i == 1:
                    self._go_to_team_select()
                else:
                    arcade.exit()

    def on_update(self, delta_time: float) -> bool | None:
        joy = self.window.joystick
        for action in self._poller.poll(joy):
            self._handle_action(action)
        return None

    def _handle_action(self, action: Action) -> None:
        if action == Action.LEFT:
            self._focus = (self._focus - 1) % 3
        elif action == Action.RIGHT:
            self._focus = (self._focus + 1) % 3
        elif action == Action.CONFIRM:
            if self._focus == 0:
                self._replay()
            elif self._focus == 1:
                self._go_to_team_select()
            else:
                arcade.exit()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _replay(self) -> None:
        from miniball.ui.match_view import MatchView

        self.window.show_view(
            MatchView(self._team_a, self._team_b, self._human_team_side)
        )

    def _go_to_team_select(self) -> None:
        from miniball.ui.team_select import TeamSelectView

        self.window.show_view(TeamSelectView())


def _draw_stat_bar_row(
    panel_x: float,
    panel_r: float,
    row_y: float,
    label: str,
    home_val: float,
    away_val: float,
    fmt: str,
) -> None:
    """Draw one statistic row: label | home value | stacked bar | away value."""
    label_w: float = 150
    val_w: float = 80
    gap: float = 10
    bar_left = panel_x + label_w + val_w + gap
    bar_right = panel_r - val_w - gap
    bar_w = bar_right - bar_left
    bar_h: float = 30

    arcade.draw_text(label, panel_x, row_y, C_HINT, 15, anchor_y="center")

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

    total = home_val + away_val
    home_frac = home_val / total if total > 0 else 0.5
    home_w = bar_w * home_frac
    bar_lo = row_y - bar_h / 2
    bar_hi = row_y + bar_h / 2

    home_fill = (C_TEAM_A[0], C_TEAM_A[1], C_TEAM_A[2], 200)
    away_fill = (C_TEAM_B[0], C_TEAM_B[1], C_TEAM_B[2], 200)

    if home_w > 0:
        arcade.draw_lrbt_rectangle_filled(
            bar_left, bar_left + home_w, bar_lo, bar_hi, home_fill
        )
    if home_w < bar_w:
        arcade.draw_lrbt_rectangle_filled(
            bar_left + home_w, bar_right, bar_lo, bar_hi, away_fill
        )

    arcade.draw_lrbt_rectangle_outline(bar_left, bar_right, bar_lo, bar_hi, C_HINT, 1)
    mid_x = bar_left + bar_w / 2
    arcade.draw_line(mid_x, bar_lo + 2, mid_x, bar_hi - 2, (160, 160, 160), 1)
