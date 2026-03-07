from __future__ import annotations

from dataclasses import dataclass, field

from miniball.ai import BaseAI


@dataclass
class PlayerConfig:
    name: str
    number: int
    start_x: float  # normalised pitch coords: 0 = left, STANDARD_PITCH_WIDTH = right
    start_y: float  # normalised pitch coords: 0 = bottom, STANDARD_PITCH_HEIGHT = top


DEFAULT_PLAYERS: list[PlayerConfig] = [
    PlayerConfig(name="GK", number=1, start_x=10, start_y=40),
    PlayerConfig(name="Defender 1", number=2, start_x=30, start_y=60),
    PlayerConfig(name="Defender 2", number=3, start_x=30, start_y=20),
    PlayerConfig(name="Forward 1", number=4, start_x=50, start_y=70),
    PlayerConfig(name="Forward 2", number=5, start_x=50, start_y=10),
]


@dataclass
class TeamConfig:
    name: str
    players: list[PlayerConfig] = field(default_factory=lambda: list(DEFAULT_PLAYERS))
    ai: BaseAI | None = None
    human_controlled: int | None = (
        None  # index into players; None = fully AI-controlled
    )
