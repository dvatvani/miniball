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
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arcade
import polars as pl

from miniball import match_stats
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
    JOY_SWITCH_THRESHOLD,
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
    global_delta_to_team,
    global_to_team,
    screen_delta_to_team,
    screen_to_team,
    team_delta_to_global,
    team_to_screen,
)
from miniball.team_config import TeamConfig

# ── Analytics types ───────────────────────────────────────────────────────────


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
        self._stats_countdown: float = (
            3.0  # delay after final whistle before showing stats
        )
        self._show_stats: bool = False
        self._team_summary_df: pl.DataFrame | None = None
        self._avg_positions_df: pl.DataFrame | None = None
        self._keys: set[int] = set()
        self._goal_flash = 0.0
        self._countdown: float = 3.0  # freeze before kick-off / after each goal

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
                """Keep a live snapshot of every axis; print significant movements."""
                game_self._joy_axis_state[axis] = value
                if abs(value) > 0.2:
                    print(f"[Joystick axis] {axis} = {value:.3f}")

        self._joy_shoot_prev = False  # edge-detect the shoot button
        self._joy_switch_prev = False  # edge-detect the right-stick player switch
        self._human_shoot_requested = (
            False  # set for one frame when human presses shoot
        )

        # Per-frame history recorded during gameplay for post-game analytics
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
            # Team B attacks left, so their normalised positions are 180°-rotated.
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

        self._ai_a: BaseAI = team_a_config.ai
        self._ai_b: BaseAI = team_b_config.ai

        # Which team list the human player belongs to (None = fully AI game).
        # Index within that list of the currently controlled player (starts on GK = 0).
        if team_a_config.human_controlled:
            self._human_team: list[Player] | None = self.team_a
        elif team_b_config.human_controlled:
            self._human_team = self.team_b
        else:
            self._human_team = None
        self._controlled_idx: int | None = 0 if self._human_team is not None else None

        # Home team (attacks left→right) always takes the opening kick-off
        self._assign_kickoff_possession(self.team_a)

    @property
    def _controlled(self) -> Player | None:
        """Human-controlled player, or ``None`` when the game is fully AI-driven."""
        if self._human_team is None or self._controlled_idx is None:
            return None
        return self._human_team[self._controlled_idx]

    @property
    def _all_players(self) -> list[Player]:
        return self.team_a + self.team_b

    # ── Drawing ──────────────────────────────────────────────────────────────

    def on_draw(self) -> None:
        self.clear()
        if self._show_stats:
            self._draw_stats_screen()
            return
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
            "Arrows / L-stick to move and aim shot · Space / A to shoot · R-stick to switch controlled player",
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
            f"{self.team_a_config.name}  {self.score_a} – {self.score_b}"
            f"  {self.team_b_config.name}",
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
            f"  {self.team_a_config.name} (home)", _PL + 14, _leg_y - 3, C_TEAM_A, 13
        )
        arcade.draw_circle_filled(_PL + _PW // 2 + 8, _leg_y + 5, 6, C_TEAM_B)
        arcade.draw_text(
            f"  {self.team_b_config.name} (away)",
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

        if self._game_over:
            if not self._show_stats:
                self._stats_countdown -= dt
                if self._stats_countdown <= 0:
                    self._show_stats = True
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
            match_df = self._build_match_df()
            if match_df is not None:
                self._export_history(match_df)
                self._team_summary_df = match_stats.team_summary(match_df)
                self._avg_positions_df = match_stats.avg_positions(match_df)
            return

        # 1. Tick all player timers
        for p in self._all_players:
            p.tick(dt)

        # 2. Gamepad inputs that are not movement: shoot button and player switch.
        #    (Movement is folded into the effective-actions layer below.)
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

        # 3. Build game states and compute effective actions (AI + human overrides).
        #    Both teams use normalised coords (attack right); flip is applied later.
        home_team_state = self._build_game_state(True)
        away_team_state = self._build_game_state(False)
        effective_a = self._get_team_effective_actions(
            self._ai_a, home_team_state, is_home_team=True
        )
        effective_b = self._get_team_effective_actions(
            self._ai_b, away_team_state, is_home_team=False
        )

        # 4. Record this frame for post-game analytics.
        controlled = self._controlled
        self._history.append(
            FrameRecord(
                game_time=GAME_DURATION - self._time_remaining,
                state=home_team_state,
                actions_team_a=effective_a,
                actions_team_b=effective_b,
                human_player=(
                    (controlled.is_home, controlled.number)
                    if controlled is not None
                    else None
                ),
            )
        )

        # 5. Apply effective actions to all players; team B vectors are flipped
        #    from normalised back to screen coordinates inside _apply_ai_actions.
        self._apply_ai_actions(self.team_a, effective_a, dt, is_home=True)
        self._apply_ai_actions(self.team_b, effective_b, dt, is_home=False)

        # Consume the single-frame shoot flag now that both teams have seen it.
        self._human_shoot_requested = False

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

    def _get_team_effective_actions(
        self,
        ai: BaseAI,
        state: GameState,
        *,
        is_home_team: bool,
    ) -> TeamActions:
        """Return the effective TeamActions for one team this frame.

        Layer 1 – AI decisions
            ``ai.get_actions(state)`` is called first to obtain the base
            actions for every player.

        Layer 2 – human overrides
            If the human-controlled player belongs to this team, their entry
            in ``directions`` is replaced with the current controller / keyboard
            input (converted from screen space to normalised coords), and the
            ``shoot`` flag is set if the human pressed the shoot button this
            frame.

        All directions in the returned dict are in normalised pitch coordinates
        (team attacks right).  The game engine's ``_apply_ai_actions`` handles
        the screen-space conversion (flip) for team B.
        """
        actions = ai.get_actions(state)

        team = self.team_a if is_home_team else self.team_b
        if self._human_team is not team or self._controlled_idx is None:
            return actions

        controlled = self._controlled
        if controlled is None:
            return actions

        # Human direction: keyboard/gamepad input is in screen space (right = +x).
        # For team B this is negated to reach their attack-right team frame.
        dx, dy = self._get_move_input()
        dx, dy = global_delta_to_team(dx, dy, is_home=is_home_team)

        # Human fully overrides the AI's action for their player: direction comes
        # from the controller, and shoot is gated solely on the button press.
        actions["actions"][controlled.number] = {
            "direction": [dx, dy],
            "shoot": self._human_shoot_requested,
        }

        return actions

    def _build_game_state(self, perspective_team_is_home: bool) -> GameState:
        """Snapshot the current game into a normalised AI state dict.

        All coordinates are flipped horizontally for Team B so that every AI
        always sees its team attacking left → right.  ``is_teammate`` is set
        to ``True`` for players on ``perspective_team``; engine-internal
        fields (``facing``, raw team label, attacking direction) are excluded.
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
            for p in self._all_players
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
            for p in self._all_players
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

    def _apply_ai_actions(
        self,
        players: list[Player],
        actions: TeamActions,
        dt: float,
        is_home: bool = True,
    ) -> None:
        """Move and optionally shoot for each player according to AI actions.

        Parameters
        ----------
        is_home:
            ``True`` for the home team (team frame = global = screen direction).
            ``False`` for the away team: both direction components are negated to
            convert from the AI's attack-right team frame back to screen space.
        """
        for p in players:
            player_action = actions["actions"].get(p.number)
            if player_action is None:
                continue
            dx, dy = player_action["direction"]
            dx, dy = team_delta_to_global(dx, dy, is_home=is_home)
            if dx != 0 or dy != 0:
                norm = math.hypot(dx, dy)
                # Allow sub-1 magnitude from analogue-style AI outputs
                speed_frac = min(norm, 1.0)
                p.x += (dx / norm) * PLAYER_SPEED * speed_frac * dt
                p.y += (dy / norm) * PLAYER_SPEED * speed_frac * dt
                p.facing = math.atan2(dy, dx)
            if player_action["shoot"]:
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
            self._trigger_goal_reset(conceding_team=self.team_a)
        elif self.ball.x - BALL_RADIUS > PITCH_R and goal_lo <= self.ball.y <= goal_hi:
            self.score_a += 1
            self._trigger_goal_reset(conceding_team=self.team_b)

    def _trigger_goal_reset(self, conceding_team: list[Player]) -> None:
        self._goal_flash = 1.5
        self.ball.reset()
        for p in self._all_players:
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
        # Sync ball position immediately (ball.update won't run during countdown)
        self.ball.update(0)

    # ── Analytics export ──────────────────────────────────────────────────────

    def _build_match_df(self) -> pl.DataFrame | None:
        """Flatten _history into a Polars DataFrame and return it.

        Each row represents one player at one frame.

        Coordinate conventions
        ──────────────────────
        ``pos_x`` / ``pos_y`` / ``action_dx`` / ``action_dy`` are in the
        team's own normalised frame (the team always attacks right, x ∈ [0,
        120], y ∈ [0, 80]).  The ``_global`` variants are in the shared pitch
        frame where team A always attacks right; for team A these are identical,
        while for team B they are the 180° rotation (W - x, H - y).
        """
        if not self._history:
            return None

        name_a = self.team_a_config.name
        name_b = self.team_b_config.name

        rows: list[dict[str, object]] = []
        for frame_number, record in enumerate(self._history):
            # Ball state is stored in the global (team A) frame.
            gbx, gby = record.state["ball"]["location"]
            gbvx, gbvy = record.state["ball"]["velocity"]
            score_a = record.state["match_state"]["team_current_score"]
            score_b = record.state["match_state"]["opposition_current_score"]
            match_time = record.state["match_state"]["match_time_seconds"]
            t = record.game_time

            _null_action: PlayerAction = {"direction": [0.0, 0.0], "shoot": False}
            hp = record.human_player  # (is_home, number) or None

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
                        "shoot": pa_a["shoot"],
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
                        "shoot": pa_b["shoot"],
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

    def _export_history(self, df: pl.DataFrame) -> None:
        """Write a match history DataFrame to a timestamped parquet file."""
        out_dir = Path("match_data")
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"match_{timestamp}.parquet"
        df.write_parquet(path)
        print(
            f"Match data saved → {path}  ({len(self._history)} frames · {len(df)} rows)"
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    from miniball.team_config import teams

    game = FootballGame(
        team_a_config=teams[0],
        team_b_config=teams[1],
    )
    game.run()


if __name__ == "__main__":
    main()
