from miniball.config import (
    PITCH_B,
    PITCH_L,
    PITCH_R,
    PITCH_T,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)


def screen_to_normalized(x: float, y: float, flip: bool = False) -> tuple[float, float]:
    scaled_x = (x - PITCH_L) / (PITCH_R - PITCH_L) * STANDARD_PITCH_WIDTH
    scaled_y = (y - PITCH_B) / (PITCH_T - PITCH_B) * STANDARD_PITCH_HEIGHT
    return (
        (scaled_x, scaled_y)
        if not flip
        else (STANDARD_PITCH_WIDTH - scaled_x, STANDARD_PITCH_HEIGHT - scaled_y)
    )


def normalized_to_screen(x: float, y: float, flip: bool = False) -> tuple[float, float]:
    scaled_x = (x * (PITCH_R - PITCH_L) / STANDARD_PITCH_WIDTH) + PITCH_L
    scaled_y = (y * (PITCH_T - PITCH_B) / STANDARD_PITCH_HEIGHT) + PITCH_B
    return (
        (scaled_x, scaled_y) if not flip else (PITCH_R - scaled_x, PITCH_T - scaled_y)
    )
