"""Backward-compatibility shim — all UI views have moved to separate modules.

This module re-exports the public names so existing imports continue to work.
Prefer importing directly from the specific modules instead.
"""

from miniball.ui.match_view import MatchView as GameView
from miniball.ui.team_select import TeamSelectView
from miniball.ui.window import MiniballWindow, main

__all__ = ["GameView", "MiniballWindow", "TeamSelectView", "main"]
