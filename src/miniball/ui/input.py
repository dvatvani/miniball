"""Unified input abstraction for the Miniball UI.

Both keyboard events and polled controller state are translated into a common
set of ``Action`` values so that view logic doesn't need to care about the
input source.
"""

from __future__ import annotations

from enum import Enum, auto

import arcade

from miniball.config import (
    STRIKE_WEIGHT_CIRCLE,
    STRIKE_WEIGHT_CROSS,
    STRIKE_WEIGHT_SQUARE,
    STRIKE_WEIGHT_TRIANGLE,
)
from miniball.ui.config import JOY_DEAD_ZONE


class Action(Enum):
    """Logical UI actions produced by any input source."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    CONFIRM = auto()
    BACK = auto()


# PlayStation button indices
_BTN_SQUARE = 0
_BTN_X = 1  # also used as menu confirm
_BTN_CIRCLE = 2  # also used as menu back
_BTN_TRIANGLE = 3

_BTN_CONFIRM = _BTN_X
_BTN_BACK = _BTN_CIRCLE

_STICK_THRESHOLD = 0.5


def actions_from_key(key: int, modifiers: int) -> list[Action]:
    """Map a keyboard key-press to zero or more logical actions."""
    if key == arcade.key.UP:
        return [Action.UP]
    if key == arcade.key.DOWN:
        return [Action.DOWN]
    if key == arcade.key.LEFT:
        return [Action.LEFT]
    if key == arcade.key.RIGHT:
        return [Action.RIGHT]
    if key == arcade.key.TAB:
        return [Action.BACK] if (modifiers & arcade.key.MOD_SHIFT) else [Action.CONFIRM]
    if key in (arcade.key.RETURN, arcade.key.SPACE):
        return [Action.CONFIRM]
    if key == arcade.key.ESCAPE:
        return [Action.BACK]
    return []


class ControllerPoller:
    """Reads a joystick once per frame and emits edge-triggered ``Action`` values.

    Call ``poll(joystick)`` each frame; it returns a list of actions that fired
    on that frame (i.e. transitions from inactive → active).
    """

    def __init__(self) -> None:
        self._ax_u = self._ax_d = False
        self._ax_l = self._ax_r = False
        self._hat_x = self._hat_y = 0
        self._btn_confirm = self._btn_back = False

    def poll(self, joy: object | None) -> list[Action]:
        """Read the controller and return newly-fired actions this frame."""
        if joy is None:
            return []

        actions: list[Action] = []

        # Left stick Y
        ay: float = -(getattr(joy, "y", 0.0) or 0.0)
        ax_u, ax_d = ay > _STICK_THRESHOLD, ay < -_STICK_THRESHOLD
        if ax_u and not self._ax_u:
            actions.append(Action.UP)
        if ax_d and not self._ax_d:
            actions.append(Action.DOWN)
        self._ax_u, self._ax_d = ax_u, ax_d

        # Left stick X
        ax: float = getattr(joy, "x", 0.0) or 0.0
        ax_r, ax_l = ax > _STICK_THRESHOLD, ax < -_STICK_THRESHOLD
        if ax_r and not self._ax_r:
            actions.append(Action.RIGHT)
        if ax_l and not self._ax_l:
            actions.append(Action.LEFT)
        self._ax_r, self._ax_l = ax_r, ax_l

        # D-pad (hat) – mirrors left stick
        hat_x = int(getattr(joy, "hat_x", 0) or 0)
        hat_y = int(getattr(joy, "hat_y", 0) or 0)
        if hat_y != self._hat_y:
            if hat_y == 1:
                actions.append(Action.UP)
            elif hat_y == -1:
                actions.append(Action.DOWN)
        if hat_x != self._hat_x:
            if hat_x == 1:
                actions.append(Action.RIGHT)
            elif hat_x == -1:
                actions.append(Action.LEFT)
        self._hat_x, self._hat_y = hat_x, hat_y

        # Buttons: X = confirm, Circle = back
        buttons = getattr(joy, "buttons", None)
        if buttons:
            btn_confirm = (
                bool(buttons[_BTN_CONFIRM]) if len(buttons) > _BTN_CONFIRM else False
            )
            btn_back = bool(buttons[_BTN_BACK]) if len(buttons) > _BTN_BACK else False
            if btn_confirm and not self._btn_confirm:
                actions.append(Action.CONFIRM)
            if btn_back and not self._btn_back:
                actions.append(Action.BACK)
            self._btn_confirm = btn_confirm
            self._btn_back = btn_back

        return actions


def get_move_vector(keys: set[int], joystick: object | None) -> tuple[float, float]:
    """Return a (dx, dy) movement vector from keyboard state and/or gamepad.

    Combines arrow keys, left analog stick, and D-pad into a single vector.
    D-pad overrides stick; stick overrides keyboard when above the dead zone.
    """
    import math

    dx = dy = 0.0

    if arcade.key.LEFT in keys:
        dx -= 1.0
    if arcade.key.RIGHT in keys:
        dx += 1.0
    if arcade.key.DOWN in keys:
        dy -= 1.0
    if arcade.key.UP in keys:
        dy += 1.0

    if joystick is not None:
        jx = getattr(joystick, "x", 0.0) or 0.0
        jy = -(getattr(joystick, "y", 0.0) or 0.0)
        if math.hypot(jx, jy) > JOY_DEAD_ZONE:
            dx, dy = jx, jy
        hat_x = int(getattr(joystick, "hat_x", 0) or 0)
        hat_y = int(getattr(joystick, "hat_y", 0) or 0)
        if hat_x != 0 or hat_y != 0:
            dx, dy = float(hat_x), float(hat_y)

    return dx, dy


def get_strike_weight(keys: set[int], joystick: object | None) -> float | None:
    """Return the strike weight if any strike button is held, else None.

    Joystick buttons take priority over keyboard.  Button-to-weight mapping:
      Square (btn 0) → STRIKE_WEIGHT_SQUARE  /  keyboard Q
      X     (btn 1) → STRIKE_WEIGHT_X        /  keyboard Space
      Circle(btn 2) → STRIKE_WEIGHT_CIRCLE   /  keyboard E
      Triangle(btn3)→ STRIKE_WEIGHT_TRIANGLE /  keyboard W
    """
    if joystick is not None:
        buttons = getattr(joystick, "buttons", None)
        if buttons:
            if len(buttons) > _BTN_SQUARE and buttons[_BTN_SQUARE]:
                return STRIKE_WEIGHT_SQUARE
            if len(buttons) > _BTN_X and buttons[_BTN_X]:
                return STRIKE_WEIGHT_CROSS
            if len(buttons) > _BTN_CIRCLE and buttons[_BTN_CIRCLE]:
                return STRIKE_WEIGHT_CIRCLE
            if len(buttons) > _BTN_TRIANGLE and buttons[_BTN_TRIANGLE]:
                return STRIKE_WEIGHT_TRIANGLE
    # Keyboard fallbacks
    if arcade.key.Q in keys:
        return STRIKE_WEIGHT_SQUARE
    if arcade.key.SPACE in keys:
        return STRIKE_WEIGHT_CROSS
    if arcade.key.E in keys:
        return STRIKE_WEIGHT_CIRCLE
    if arcade.key.W in keys:
        return STRIKE_WEIGHT_TRIANGLE
    return None
