from __future__ import annotations

from dataclasses import dataclass

from miniball.ai import BaseAI, StationaryAI
from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH


@dataclass
class PlayerConfig:
    name: str
    number: int
    x: float  # normalised pitch coords: 0 = left, STANDARD_PITCH_WIDTH = right
    y: float  # normalised pitch coords: 0 = bottom, STANDARD_PITCH_HEIGHT = top


DEFAULT_PLAYERS: list[PlayerConfig] = [
    PlayerConfig(name="GK", number=1, x=10, y=40),
    PlayerConfig(name="Defender 1", number=2, x=50, y=60),
    PlayerConfig(name="Defender 2", number=3, x=50, y=20),
    PlayerConfig(name="Forward 1", number=4, x=100, y=70),
    PlayerConfig(name="Forward 2", number=5, x=100, y=10),
]


class TeamConfig:
    def __init__(
        self,
        name: str,
        players: list[PlayerConfig] | None = None,
        ai: type[BaseAI] = StationaryAI,
        human_controlled: int | None = None,
    ) -> None:
        self.name = name
        self.players = players if players is not None else list(DEFAULT_PLAYERS)
        formation = {p.name: [p.x, p.y] for p in self.players}
        self.ai = ai(formation=formation)
        self.human_controlled = human_controlled
        assert len(self.players) == 5, "Team must have 5 players"
        assert all(isinstance(p, PlayerConfig) for p in self.players), (
            "Players must be of type PlayerConfig"
        )
        assert all(p.name is not None for p in self.players), "Players must have a name"
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
