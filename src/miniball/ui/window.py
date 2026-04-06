"""Root window and entry point for the Miniball game UI."""

from __future__ import annotations

import arcade

from miniball.ui.config import SCREEN_H, SCREEN_W, TITLE


class MiniballWindow(arcade.Window):
    """Root window that owns the joystick and hosts the active view."""

    def __init__(self) -> None:
        super().__init__(SCREEN_W, SCREEN_H, TITLE)
        arcade.set_background_color((30, 30, 30, 255))

        joysticks = arcade.get_joysticks()  # type: ignore[attr-defined]
        self.joystick = joysticks[0] if joysticks else None
        self.joy_axis_state: dict[str, float] = {}

        if self.joystick is not None:
            self.joystick.open()
            win = self

            @self.joystick.event
            def on_joyaxis_motion(joystick, axis: str, value: float) -> None:  # noqa: ANN001
                """Keep a live snapshot of every axis."""
                win.joy_axis_state[axis] = value


def main() -> None:
    """Launch the game: show the team-selection screen and start the event loop."""
    from miniball.ui.team_select import TeamSelectView

    window = MiniballWindow()
    window.show_view(TeamSelectView())
    arcade.run()
