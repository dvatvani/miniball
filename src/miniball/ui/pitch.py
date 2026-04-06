"""Shared pitch-drawing helper used by both the live match and stats views."""

from __future__ import annotations

import arcade

from miniball.config import (
    STANDARD_GOAL_DEPTH,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)
from miniball.ui.config import (
    C_BORDER_AWAY,
    C_BORDER_HOME,
    C_GOAL_AWAY,
    C_GOAL_HOME,
    C_PITCH_AWAY,
    C_PITCH_HOME,
)


def draw_pitch(
    left: float,
    bottom: float,
    width: float,
    height: float,
    line_width: int = 2,
) -> None:
    """Draw the two-tone pitch with goals, boundaries, and centre line.

    Parameters
    ----------
    left, bottom:
        Screen-pixel origin of the pitch rectangle (bottom-left corner).
    width, height:
        Screen-pixel dimensions of the playing field.
    line_width:
        Pixel thickness for boundary and goal lines.
    """
    right = left + width
    top = bottom + height
    cx = left + width / 2
    cy = bottom + height / 2

    goal_h = height * (STANDARD_GOAL_HEIGHT / STANDARD_PITCH_HEIGHT)
    goal_d = width * (STANDARD_GOAL_DEPTH / STANDARD_PITCH_WIDTH)
    goal_lo = cy - goal_h / 2
    goal_hi = cy + goal_h / 2

    # Pitch halves
    arcade.draw_lrbt_rectangle_filled(left, cx, bottom, top, C_PITCH_HOME)
    arcade.draw_lrbt_rectangle_filled(cx, right, bottom, top, C_PITCH_AWAY)

    # Goal fills
    arcade.draw_lrbt_rectangle_filled(
        left - goal_d, left, goal_lo, goal_hi, C_GOAL_HOME
    )
    arcade.draw_lrbt_rectangle_filled(
        right, right + goal_d, goal_lo, goal_hi, C_GOAL_AWAY
    )

    # Home (left) boundary + goal outline
    arcade.draw_line(left, bottom, left, top, C_BORDER_HOME, line_width)
    arcade.draw_line(left, top, cx, top, C_BORDER_HOME, line_width)
    arcade.draw_line(left, bottom, cx, bottom, C_BORDER_HOME, line_width)
    arcade.draw_line(
        left - goal_d, goal_lo, left - goal_d, goal_hi, C_BORDER_HOME, line_width
    )
    arcade.draw_line(left - goal_d, goal_lo, left, goal_lo, C_BORDER_HOME, line_width)
    arcade.draw_line(left - goal_d, goal_hi, left, goal_hi, C_BORDER_HOME, line_width)

    # Away (right) boundary + goal outline
    arcade.draw_line(right, bottom, right, top, C_BORDER_AWAY, line_width)
    arcade.draw_line(cx, top, right, top, C_BORDER_AWAY, line_width)
    arcade.draw_line(cx, bottom, right, bottom, C_BORDER_AWAY, line_width)
    arcade.draw_line(
        right + goal_d, goal_lo, right + goal_d, goal_hi, C_BORDER_AWAY, line_width
    )
    arcade.draw_line(right, goal_lo, right + goal_d, goal_lo, C_BORDER_AWAY, line_width)
    arcade.draw_line(right, goal_hi, right + goal_d, goal_hi, C_BORDER_AWAY, line_width)
