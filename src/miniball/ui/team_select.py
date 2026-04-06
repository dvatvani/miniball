"""Team-selection screen: choose home/away teams and which side to play."""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import arcade

from miniball.teams import teams_list
from miniball.ui.config import C_CONTROLLED, C_HINT, C_HUD, SCREEN_W
from miniball.ui.input import Action, ControllerPoller, actions_from_key

# ── Layout constants ──────────────────────────────────────────────────────────

HOME_X = 300
AWAY_X = 900
HEADER_Y = 672
LIST_TOP = 640
ITEM_H = 50
ITEM_HW = 140
ITEM_HH = 20
MAX_VISIBLE = 6
SIDE_LABEL_Y = 342
SIDE_Y = 296
SIDE_CENTERS = (400, 600, 800)
SIDE_BTN_HW = 95
SIDE_BTN_HH = 22
START_CX = 600
START_CY = 208
START_HW = 130
START_HH = 28
HINT_Y = 152


class _Focus(IntEnum):
    HOME = 0
    AWAY = 1
    SIDE = 2
    START = 3


class TeamSelectView(arcade.View):
    """Pre-game lobby screen for choosing teams and player side."""

    def __init__(self) -> None:
        super().__init__()
        self._teams = teams_list
        self._home_idx = 0
        self._away_idx = 1 if len(self._teams) > 1 else 0
        self._side_idx = 2  # 0 = home, 1 = away, 2 = watch

        self._focus: _Focus = _Focus.HOME
        self._home_scroll = 0
        self._away_scroll = 0
        self._poller = ControllerPoller()

    def on_show_view(self) -> None:
        arcade.set_background_color((30, 30, 30, 255))

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        self._draw_title()
        self._draw_team_lists()
        self._draw_side_selector()
        self._draw_start_button()
        arcade.draw_text(
            "Arrows/D-pad: navigate   ·   Tab/X: advance section   ·   Shift-Tab/○: back   ·   Enter/Space: kick off",
            SCREEN_W / 2,
            HINT_Y,
            C_HINT,
            13,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_title(self) -> None:
        arcade.draw_text(
            "MINIBALL",
            SCREEN_W / 2,
            755,
            C_HUD,
            52,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        arcade.draw_text(
            "Select Teams",
            SCREEN_W / 2,
            710,
            C_HINT,
            18,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_team_lists(self) -> None:
        n = len(self._teams)
        for col_x, col_focus, sel_idx, scroll in (
            (HOME_X, _Focus.HOME, self._home_idx, self._home_scroll),
            (AWAY_X, _Focus.AWAY, self._away_idx, self._away_scroll),
        ):
            focused = self._focus == col_focus
            label = "HOME" if col_focus == _Focus.HOME else "AWAY"
            hdr_color = C_CONTROLLED if focused else C_HUD
            arcade.draw_text(
                label,
                col_x,
                HEADER_Y,
                hdr_color,
                20,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
            if focused:
                arcade.draw_line(
                    col_x - 38,
                    HEADER_Y - 14,
                    col_x + 38,
                    HEADER_Y - 14,
                    C_CONTROLLED,
                    2,
                )

            visible = min(MAX_VISIBLE, n)
            for vis_pos in range(visible):
                i = scroll + vis_pos
                if i >= n:
                    break
                team = self._teams[i]
                iy = LIST_TOP - vis_pos * ITEM_H
                is_sel = i == sel_idx
                if is_sel:
                    bg = (40, 60, 120) if focused else (50, 50, 80)
                    arcade.draw_lrbt_rectangle_filled(
                        col_x - ITEM_HW,
                        col_x + ITEM_HW,
                        iy - ITEM_HH,
                        iy + ITEM_HH,
                        bg,
                    )
                    if focused:
                        arcade.draw_lrbt_rectangle_outline(
                            col_x - ITEM_HW,
                            col_x + ITEM_HW,
                            iy - ITEM_HH,
                            iy + ITEM_HH,
                            C_CONTROLLED,
                            2,
                        )
                txt = (
                    C_CONTROLLED
                    if (is_sel and focused)
                    else C_HUD
                    if is_sel
                    else C_HINT
                )
                arcade.draw_text(
                    team.name,
                    col_x,
                    iy,
                    txt,
                    16,
                    anchor_x="center",
                    anchor_y="center",
                )

            # Scrollbar
            if n > MAX_VISIBLE:
                sb_x = col_x + ITEM_HW + 10
                track_top = LIST_TOP + ITEM_HH
                track_bot = LIST_TOP - (visible - 1) * ITEM_H - ITEM_HH
                track_h = track_top - track_bot
                arcade.draw_line(sb_x, track_bot, sb_x, track_top, (60, 60, 80), 3)
                thumb_h = max(16.0, track_h * MAX_VISIBLE / n)
                max_scroll = n - MAX_VISIBLE
                thumb_bot = (
                    track_top - thumb_h - (scroll / max_scroll) * (track_h - thumb_h)
                )
                arcade.draw_line(
                    sb_x,
                    thumb_bot,
                    sb_x,
                    thumb_bot + thumb_h,
                    C_CONTROLLED if focused else C_HINT,
                    4,
                )

    def _draw_side_selector(self) -> None:
        focused = self._focus == _Focus.SIDE
        arcade.draw_text(
            "Play as:",
            SCREEN_W / 2,
            SIDE_LABEL_Y,
            C_CONTROLLED if focused else C_HUD,
            18,
            anchor_x="center",
            anchor_y="center",
        )
        labels = ("Home", "Away", "Watch")
        for j, (bx, lbl) in enumerate(zip(SIDE_CENTERS, labels)):
            is_sel = j == self._side_idx
            bg = (50, 90, 155) if is_sel else (38, 38, 55)
            border = (
                C_CONTROLLED
                if (focused and is_sel)
                else C_HUD
                if is_sel
                else (80, 80, 100)
            )
            arcade.draw_lrbt_rectangle_filled(
                bx - SIDE_BTN_HW,
                bx + SIDE_BTN_HW,
                SIDE_Y - SIDE_BTN_HH,
                SIDE_Y + SIDE_BTN_HH,
                bg,
            )
            arcade.draw_lrbt_rectangle_outline(
                bx - SIDE_BTN_HW,
                bx + SIDE_BTN_HW,
                SIDE_Y - SIDE_BTN_HH,
                SIDE_Y + SIDE_BTN_HH,
                border,
                2,
            )
            arcade.draw_text(
                lbl,
                bx,
                SIDE_Y,
                C_HUD if is_sel else C_HINT,
                16,
                anchor_x="center",
                anchor_y="center",
            )

    def _draw_start_button(self) -> None:
        focused = self._focus == _Focus.START
        border = C_CONTROLLED if focused else (120, 195, 130)
        arcade.draw_lrbt_rectangle_filled(
            START_CX - START_HW,
            START_CX + START_HW,
            START_CY - START_HH,
            START_CY + START_HH,
            (40, 110, 50),
        )
        arcade.draw_lrbt_rectangle_outline(
            START_CX - START_HW,
            START_CX + START_HW,
            START_CY - START_HH,
            START_CY + START_HH,
            border,
            2,
        )
        arcade.draw_text(
            "KICK OFF",
            START_CX,
            START_CY,
            C_HUD,
            22,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    # ── Input ─────────────────────────────────────────────────────────────────

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        for action in actions_from_key(symbol, modifiers):
            self._handle_action(action)
        return None

    def on_mouse_scroll(
        self, x: float, y: float, scroll_x: float, scroll_y: float
    ) -> None:
        """Scroll the list under the mouse cursor with the mouse wheel."""
        n = len(self._teams)
        if n <= MAX_VISIBLE:
            return
        delta = -int(scroll_y)
        if abs(x - HOME_X) <= ITEM_HW:
            self._home_scroll = max(0, min(n - MAX_VISIBLE, self._home_scroll + delta))
        elif abs(x - AWAY_X) <= ITEM_HW:
            self._away_scroll = max(0, min(n - MAX_VISIBLE, self._away_scroll + delta))

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        for col_x, col_focus, other_idx, scroll in (
            (HOME_X, _Focus.HOME, self._away_idx, self._home_scroll),
            (AWAY_X, _Focus.AWAY, self._home_idx, self._away_scroll),
        ):
            for vis_pos in range(min(MAX_VISIBLE, len(self._teams))):
                i = scroll + vis_pos
                if i >= len(self._teams):
                    break
                iy = LIST_TOP - vis_pos * ITEM_H
                if (
                    col_x - ITEM_HW <= x <= col_x + ITEM_HW
                    and iy - ITEM_HH <= y <= iy + ITEM_HH
                    and i != other_idx
                ):
                    if col_focus == _Focus.HOME:
                        self._home_idx = i
                    else:
                        self._away_idx = i
                    self._focus = col_focus

        for j, bx in enumerate(SIDE_CENTERS):
            if (
                bx - SIDE_BTN_HW <= x <= bx + SIDE_BTN_HW
                and SIDE_Y - SIDE_BTN_HH <= y <= SIDE_Y + SIDE_BTN_HH
            ):
                self._side_idx = j
                self._focus = _Focus.SIDE

        if abs(x - START_CX) <= START_HW and abs(y - START_CY) <= START_HH:
            self._start_game()

    def on_update(self, delta_time: float) -> bool | None:
        joy = self.window.joystick
        for action in self._poller.poll(joy):
            self._handle_action(action)
        return None

    # ── Unified action handler ────────────────────────────────────────────────

    def _handle_action(self, action: Action) -> None:
        if action == Action.UP:
            self._navigate_ud(-1)
        elif action == Action.DOWN:
            self._navigate_ud(1)
        elif action == Action.LEFT:
            if self._focus == _Focus.SIDE:
                self._side_idx = (self._side_idx - 1) % 3
        elif action == Action.RIGHT:
            if self._focus == _Focus.SIDE:
                self._side_idx = (self._side_idx + 1) % 3
        elif action == Action.CONFIRM:
            if self._focus == _Focus.START:
                self._start_game()
            else:
                self._focus = _Focus((int(self._focus) + 1) % 4)
        elif action == Action.BACK:
            if self._focus != _Focus.HOME:
                self._focus = _Focus((int(self._focus) - 1) % 4)

    # ── Navigation helpers ────────────────────────────────────────────────────

    def _navigate_ud(self, delta: int) -> None:
        """Move selection up/down within the focused team list."""
        n = len(self._teams)
        if self._focus == _Focus.HOME:
            new = (self._home_idx + delta) % n
            if new == self._away_idx:
                new = (new + delta) % n
            self._home_idx = new
            self._home_scroll = self._clamp_scroll(new, self._home_scroll, n)
        elif self._focus == _Focus.AWAY:
            new = (self._away_idx + delta) % n
            if new == self._home_idx:
                new = (new + delta) % n
            self._away_idx = new
            self._away_scroll = self._clamp_scroll(new, self._away_scroll, n)

    def _clamp_scroll(self, sel: int, scroll: int, n: int) -> int:
        """Return a scroll offset that keeps ``sel`` within the visible window."""
        max_scroll = max(0, n - MAX_VISIBLE)
        if sel < scroll:
            return sel
        if sel >= scroll + MAX_VISIBLE:
            return min(sel - MAX_VISIBLE + 1, max_scroll)
        return scroll

    def _start_game(self) -> None:
        from miniball.ui.match_view import MatchView

        home = self._teams[self._home_idx]
        away = self._teams[self._away_idx]
        side_map: dict[int, Literal["home", "away"] | None] = {
            0: "home",
            1: "away",
            2: None,
        }
        self.window.show_view(
            MatchView(home, away, human_team=side_map[self._side_idx])
        )
