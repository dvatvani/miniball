from __future__ import annotations

import re
from dataclasses import dataclass

from miniball.ai import BallChasersAI, BaseAI, BaselineAI, StationaryAI
from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH

# Characters that are illegal or problematic in file paths on macOS, Linux, or Windows.
_UNSAFE_FILENAME_RE = re.compile(r'[/\\:*?"<>|\x00\r\n]')


@dataclass
class FormationSlot:
    """Starting position and shirt number for one player in a team definition."""

    number: int
    x: float  # normalised pitch coords: 0 = left, STANDARD_PITCH_WIDTH = right
    y: float  # normalised pitch coords: 0 = bottom, STANDARD_PITCH_HEIGHT = top


DEFAULT_PLAYERS: list[FormationSlot] = [
    FormationSlot(number=1, x=5, y=40),
    FormationSlot(number=2, x=50, y=60),
    FormationSlot(number=3, x=50, y=20),
    FormationSlot(number=4, x=100, y=50),
    FormationSlot(number=5, x=100, y=30),
]


class Team:
    def __init__(
        self,
        name: str,
        players: list[FormationSlot] | None = None,
        ai: type[BaseAI] = StationaryAI,
    ) -> None:
        self.name = name
        self.players = players if players is not None else list(DEFAULT_PLAYERS)
        formation = {p.number: (p.x, p.y) for p in self.players}
        self.ai = ai(formation=formation)
        assert len(self.players) == 5, "Team must have 5 players"
        assert all(isinstance(p, FormationSlot) for p in self.players), (
            "Players must be of type FormationSlot"
        )
        assert all(p.number is not None for p in self.players), (
            "Players must have a number"
        )
        assert all(p.x is not None for p in self.players), (
            "Players must have an x position"
        )
        assert all(p.y is not None for p in self.players), (
            "Players must have a y position"
        )
        assert all(0 <= p.x <= STANDARD_PITCH_WIDTH for p in self.players), (
            "Players must have an x position between 0 and STANDARD_PITCH_WIDTH"
        )
        assert all(0 <= p.y <= STANDARD_PITCH_HEIGHT for p in self.players), (
            "Players must have a y position between 0 and STANDARD_PITCH_HEIGHT"
        )
        assert len(set(p.number for p in self.players)) == len(self.players), (
            "Players must have unique numbers"
        )
        assert len(name) <= 30, "Team name must be 30 characters or fewer"
        assert not _UNSAFE_FILENAME_RE.search(name), (
            f"Team name {name!r} contains invalid characters"
            r"(forbidden: / \ : * ? \" < > | and control characters)"
        )


teams_list = [
    Team(name="Baseline (1-2-2)", ai=BaselineAI),
    Team(
        name="Baseline (1-3-1)",
        ai=BaselineAI,
        players=[
            FormationSlot(number=1, x=5, y=40),
            FormationSlot(number=2, x=40, y=60),
            FormationSlot(number=3, x=30, y=40),
            FormationSlot(number=4, x=40, y=20),
            FormationSlot(number=5, x=105, y=40),
        ],
    ),
    Team(
        name="Baseline (1-1-3)",
        ai=BaselineAI,
        players=[
            FormationSlot(number=1, x=5, y=40),
            FormationSlot(number=2, x=40, y=40),
            FormationSlot(number=3, x=100, y=40),
            FormationSlot(number=4, x=105, y=60),
            FormationSlot(number=5, x=105, y=20),
        ],
    ),
    Team(name="Ball Chasers", ai=BallChasersAI),
    Team(name="Stationary", ai=StationaryAI),
    Team(
        name="Static Defensive",
        ai=StationaryAI,
        players=[
            FormationSlot(number=1, x=5, y=40),
            FormationSlot(number=2, x=15, y=55),
            FormationSlot(number=3, x=15, y=45),
            FormationSlot(number=4, x=15, y=35),
            FormationSlot(number=5, x=15, y=25),
        ],
    ),
    Team(name="Stationary2", ai=StationaryAI),
    Team(name="Stationary3", ai=StationaryAI),
    Team(name="Stationary4", ai=StationaryAI),
    Team(name="Stationary5", ai=StationaryAI),
]

teams = {team.name: team for team in teams_list}
