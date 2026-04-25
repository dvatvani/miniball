"""Post-match statistics screen with action buttons."""

from __future__ import annotations

from typing import Literal, cast

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

# ── Stats area geometry ───────────────────────────────────────────────────────

# Outer border of the selectable stats area (all pages)
_STATS_L = 25
_STATS_R = 1175
_STATS_T = 690  # top of stats area (below score header)
_STATS_B = 155  # bottom of stats area (above buttons)

# Page tab strip (inside the stats area, at the top)
_TAB_Y = 673  # tab-centre y
_TAB_H = 22  # tab height
_TAB_W = 155  # tab width
_TAB_GAP = 8  # gap between adjacent tabs

# Top of page content (below tab strip + small gap)
_CONTENT_TOP = 653

# ── Player stats table geometry (pages 1 & 2) ─────────────────────────────────

_TABLE_L = 80  # left edge of row backgrounds
_TABLE_R = 1120  # right edge of row backgrounds
_TABLE_ROW_H = 34

# Column centres: 5 equal slices of the 1040 px table width
_COL_KEYS = ("#", "P", "BR", "T", "G")
_COL_CX: dict[str, float] = {
    k: _TABLE_L + (i + 0.5) * (_TABLE_R - _TABLE_L) / len(_COL_KEYS)
    for i, k in enumerate(_COL_KEYS)
}

# Colour-coding targets
_C_GOOD = (50, 210, 90)  # green – high positive stats
_C_BAD = (220, 60, 60)  # red   – high negative stats
_C_HUMAN_BG = (180, 150, 20, 90)  # amber – human reference row background


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
        player_match_df: pl.DataFrame | None = None,
        human_player_match_df: pl.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self._team_a = team_a
        self._team_b = team_b
        self._score_a = score_a
        self._score_b = score_b
        self._human_team_side = human_team_side
        self._team_summary_df = team_summary_df
        self._avg_positions_df = avg_positions_df
        self._player_match_df = player_match_df
        self._human_player_match_df = human_player_match_df

        # Build ordered page list
        self._pages: list[str] = ["Overview"]
        if player_match_df is not None:
            self._pages.append("Player Stats")
        if human_player_match_df is not None and human_team_side is not None:
            self._pages.append("Your Stats")
        self._stats_page: int = 0

        # Focus: -1 = stats area, 0/1/2 = buttons (Replay / Team Select / Quit)
        self._focus: int = 0
        self._poller = ControllerPoller()

    def on_show_view(self) -> None:
        arcade.set_background_color((30, 30, 30, 255))

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        self._draw_header()
        self._draw_stats_area()
        self._draw_nav_hint()
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

    def _draw_stats_area(self) -> None:
        self._draw_page_tabs()

        if self._stats_page == 0:
            self._draw_mini_pitch()
            self._draw_stats_panel()
        elif self._stats_page == 1:
            self._draw_player_stats_table()
        elif self._stats_page == 2:
            self._draw_human_stats_table()

    def _draw_page_tabs(self) -> None:
        """Render the row of page-selector tabs at the top of the stats area."""
        n = len(self._pages)
        total_w = n * _TAB_W + (n - 1) * _TAB_GAP
        x0 = SCREEN_W / 2 - total_w / 2

        for i, name in enumerate(self._pages):
            tx = x0 + i * (_TAB_W + _TAB_GAP)
            cx = tx + _TAB_W / 2
            is_current = i == self._stats_page
            stats_focused = self._focus == -1

            if is_current:
                bg = (70, 65, 20) if stats_focused else (55, 55, 75)
                border = C_CONTROLLED if stats_focused else (120, 120, 160)
                tc = (255, 230, 80) if stats_focused else C_HUD
            else:
                bg = (35, 35, 50)
                border = (65, 65, 85)
                tc = C_HINT

            arcade.draw_lrbt_rectangle_filled(
                tx,
                tx + _TAB_W,
                _TAB_Y - _TAB_H / 2,
                _TAB_Y + _TAB_H / 2,
                bg,
            )
            arcade.draw_lrbt_rectangle_outline(
                tx,
                tx + _TAB_W,
                _TAB_Y - _TAB_H / 2,
                _TAB_Y + _TAB_H / 2,
                border,
                1,
            )
            arcade.draw_text(
                name,
                cx,
                _TAB_Y,
                tc,
                13,
                anchor_x="center",
                anchor_y="center",
                bold=is_current,
            )

    def _draw_nav_hint(self) -> None:
        """Small navigation hint in the gap between stats area and buttons."""
        if self._focus == -1:
            hint = "← / → to switch views   ·   ↓ to select buttons"
        else:
            hint = "↑ to navigate views"
        arcade.draw_text(
            hint,
            SCREEN_W / 2,
            136,
            C_HINT,
            11,
            anchor_x="center",
            anchor_y="center",
        )

    # ── Page 0: Overview ──────────────────────────────────────────────────────

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
            ("Goals", _fval(home["team_score"]), _fval(away["team_score"]), "{:.0f}"),
            (
                "Strikes attempted",
                _fval(home["strikes_attempted"]),
                _fval(away["strikes_attempted"]),
                "{:.0f}",
            ),
            (
                "Strikes completed",
                _fval(home["strikes_successful"]),
                _fval(away["strikes_successful"]),
                "{:.0f}",
            ),
            (
                "Strike conversion",
                _fval(home["strike_completion_rate"]),
                _fval(away["strike_completion_rate"]),
                "{:.1f}%",
            ),
            (
                "Tackles",
                _fval(home["tackles"]),
                _fval(away["tackles"]),
                "{:.0f}",
            ),
            (
                "Interceptions",
                _fval(home["interceptions"]),
                _fval(away["interceptions"]),
                "{:.0f}",
            ),
            (
                "Possession",
                _fval(home["possession_percentage"]),
                _fval(away["possession_percentage"]),
                "{:.1f}%",
            ),
        ]
        n = len(stat_rows)
        step = (top_row_y - bot_row_y) / (n - 1) if n > 1 else 0.0
        for i, (label, h_val, a_val, fmt) in enumerate(stat_rows):
            _draw_stat_bar_row(sx, sr, top_row_y - i * step, label, h_val, a_val, fmt)

    # ── Page 1: Player stats table ────────────────────────────────────────────

    def _draw_player_stats_table(self) -> None:
        if self._player_match_df is None:
            return
        df = self._player_match_df

        # Title
        title_y = _CONTENT_TOP - 15
        arcade.draw_text(
            "PLAYER STATS",
            SCREEN_W / 2,
            title_y,
            C_HUD,
            20,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        # Column headers
        hdr_y = title_y - 32
        for key, cx in _COL_CX.items():
            arcade.draw_text(
                key,
                cx,
                hdr_y,
                C_HINT,
                14,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

        sep_y = hdr_y - 14
        arcade.draw_line(_TABLE_L, sep_y, _TABLE_R, sep_y, (70, 70, 70), 1)

        # Max values across all players for colour coding
        max_p = max(1, cast(int, df["strikes_successful"].max() or 1))
        max_br = max(1, cast(int, df["ball_recoveries"].max() or 1))
        max_t = max(1, cast(int, df["turnovers"].max() or 1))
        max_g = max(1, cast(int, df["goals"].max() or 1))

        home_rows = df.filter(pl.col("is_home")).sort("player_number")
        away_rows = df.filter(~pl.col("is_home")).sort("player_number")
        all_rows = pl.concat([home_rows, away_rows])

        row_y = sep_y - _TABLE_ROW_H / 2 - 5

        for row in all_rows.iter_rows(named=True):
            team_color: tuple[int, int, int] = C_TEAM_A if row["is_home"] else C_TEAM_B
            _draw_player_row(
                row_y,
                str(row["player_number"]),
                int(row["strikes_successful"] or 0),
                int(row["ball_recoveries"] or 0),
                int(row["turnovers"] or 0),
                int(row["goals"] or 0),
                team_color=team_color,
                max_p=max_p,
                max_br=max_br,
                max_t=max_t,
                max_g=max_g,
            )
            row_y -= _TABLE_ROW_H

        # Human reference row (with extra gap)
        if self._human_player_match_df is not None:
            hdf = self._human_player_match_df
            h_p = int(hdf["strikes_successful"].sum() or 0)
            h_br = int(hdf["ball_recoveries"].sum() or 0)
            h_t = int(hdf["turnovers"].sum() or 0)
            h_g = int(hdf["goals"].sum() or 0)
            human_row_y = row_y - 20  # extra spacing before human row
            _draw_player_row(
                human_row_y,
                "YOU",
                h_p,
                h_br,
                h_t,
                h_g,
                team_color=None,  # None → yellow human style
                max_p=max_p,
                max_br=max_br,
                max_t=max_t,
                max_g=max_g,
            )

    # ── Page 2: Human-controlled stats ────────────────────────────────────────

    def _draw_human_stats_table(self) -> None:
        if self._human_player_match_df is None:
            return
        df = self._human_player_match_df

        # Title
        title_y = _CONTENT_TOP - 15
        arcade.draw_text(
            "YOUR STATS",
            SCREEN_W / 2,
            title_y,
            (255, 220, 0),
            20,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        arcade.draw_text(
            "Stats from frames when you were in control of each player",
            SCREEN_W / 2,
            title_y - 24,
            C_HINT,
            12,
            anchor_x="center",
            anchor_y="center",
        )

        # Column headers
        hdr_y = title_y - 54
        for key, cx in _COL_CX.items():
            arcade.draw_text(
                key,
                cx,
                hdr_y,
                C_HINT,
                14,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

        sep_y = hdr_y - 14
        arcade.draw_line(_TABLE_L, sep_y, _TABLE_R, sep_y, (70, 70, 70), 1)

        # Max values relative to this table only
        max_p = max(1, cast(int, df["strikes_successful"].max() or 1))
        max_br = max(1, cast(int, df["ball_recoveries"].max() or 1))
        max_t = max(1, cast(int, df["turnovers"].max() or 1))
        max_g = max(1, cast(int, df["goals"].max() or 1))

        row_y = sep_y - _TABLE_ROW_H / 2 - 5

        for row in df.sort("player_number").iter_rows(named=True):
            _draw_player_row(
                row_y,
                str(row["player_number"]),
                int(row["strikes_successful"] or 0),
                int(row["ball_recoveries"] or 0),
                int(row["turnovers"] or 0),
                int(row["goals"] or 0),
                team_color=None,  # all yellow on this page
                max_p=max_p,
                max_br=max_br,
                max_t=max_t,
                max_g=max_g,
            )
            row_y -= _TABLE_ROW_H

    # ── Buttons ───────────────────────────────────────────────────────────────

    def _draw_buttons(self) -> None:
        for i, (cx, label) in enumerate(zip(BTN_CXS, BTN_LABELS)):
            focused = i == self._focus  # False when _focus == -1
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
        # Tab clicks — switch page and focus the stats area
        n = len(self._pages)
        total_w = n * _TAB_W + (n - 1) * _TAB_GAP
        x0 = SCREEN_W / 2 - total_w / 2
        for i in range(n):
            tx = x0 + i * (_TAB_W + _TAB_GAP)
            if tx <= x <= tx + _TAB_W and abs(y - _TAB_Y) <= _TAB_H / 2:
                self._stats_page = i
                self._focus = -1
                return

        # Action button clicks
        for i, cx in enumerate(BTN_CXS):
            if abs(x - cx) <= BTN_HW and abs(y - BTN_CY) <= BTN_HH:
                if i == 0:
                    self._replay()
                elif i == 1:
                    self._go_to_team_select()
                else:
                    arcade.exit()
                return

        # Click anywhere else in the stats area focuses it
        if _STATS_L <= x <= _STATS_R and _STATS_B <= y <= _STATS_T + 13:
            self._focus = -1

    def on_update(self, delta_time: float) -> bool | None:
        joy = self.window.joystick
        for action in self._poller.poll(joy):
            self._handle_action(action)
        return None

    def _handle_action(self, action: Action) -> None:
        if self._focus == -1:
            # Stats area is focused: LEFT/RIGHT switches pages
            if action == Action.LEFT:
                self._stats_page = (self._stats_page - 1) % len(self._pages)
            elif action == Action.RIGHT:
                self._stats_page = (self._stats_page + 1) % len(self._pages)
            elif action in (Action.DOWN, Action.CONFIRM):
                self._focus = 0
        else:
            # Button focused: LEFT/RIGHT moves between buttons; UP focuses stats
            if action == Action.LEFT:
                self._focus = (self._focus - 1) % 3
            elif action == Action.RIGHT:
                self._focus = (self._focus + 1) % 3
            elif action == Action.UP:
                self._focus = -1
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


# ── Module-level helpers ──────────────────────────────────────────────────────


def _stat_color(value: int, max_val: int, is_bad: bool) -> tuple[int, int, int]:
    """Interpolate between grey and a target colour based on value / max_val."""
    intensity = value / max_val if max_val > 0 else 0.0
    base = C_HINT  # (180, 180, 180)
    target = _C_BAD if is_bad else _C_GOOD
    r = int(base[0] + intensity * (target[0] - base[0]))
    g = int(base[1] + intensity * (target[1] - base[1]))
    b = int(base[2] + intensity * (target[2] - base[2]))
    return (r, g, b)


def _draw_player_row(
    row_y: float,
    label: str,
    p: int,
    br: int,
    t: int,
    g: int,
    team_color: tuple[int, int, int] | None,
    max_p: int,
    max_br: int,
    max_t: int,
    max_g: int,
) -> None:
    """Draw one player row: background, number/label, and colour-coded stats."""
    row_lo = row_y - _TABLE_ROW_H / 2
    row_hi = row_y + _TABLE_ROW_H / 2

    if team_color is None:
        bg: tuple[int, int, int, int] = _C_HUMAN_BG
        label_color: tuple[int, int, int] = (255, 230, 0)
    else:
        bg = (team_color[0], team_color[1], team_color[2], 70)
        label_color = (210, 210, 210)

    arcade.draw_lrbt_rectangle_filled(_TABLE_L, _TABLE_R, row_lo, row_hi, bg)

    arcade.draw_text(
        label,
        _COL_CX["#"],
        row_y,
        label_color,
        14,
        anchor_x="center",
        anchor_y="center",
        bold=True,
    )
    arcade.draw_text(
        str(p),
        _COL_CX["P"],
        row_y,
        _stat_color(p, max_p, False),
        14,
        anchor_x="center",
        anchor_y="center",
    )
    arcade.draw_text(
        str(br),
        _COL_CX["BR"],
        row_y,
        _stat_color(br, max_br, False),
        14,
        anchor_x="center",
        anchor_y="center",
    )
    arcade.draw_text(
        str(t),
        _COL_CX["T"],
        row_y,
        _stat_color(t, max_t, True),
        14,
        anchor_x="center",
        anchor_y="center",
    )
    arcade.draw_text(
        str(g),
        _COL_CX["G"],
        row_y,
        _stat_color(g, max_g, False),
        14,
        anchor_x="center",
        anchor_y="center",
    )


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
