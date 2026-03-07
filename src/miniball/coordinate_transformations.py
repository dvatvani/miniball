from miniball.config import (
    SCREEN_H,
    SCREEN_W,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)


def screen_to_normalized(x: float, y: float, flip: bool = False) -> tuple[float, float]:
    scaled_x = x / SCREEN_W * STANDARD_PITCH_WIDTH
    scaled_y = y / SCREEN_H * STANDARD_PITCH_HEIGHT
    return (
        (scaled_x, scaled_y)
        if not flip
        else (STANDARD_PITCH_WIDTH - scaled_x, STANDARD_PITCH_HEIGHT - scaled_y)
    )


def normalized_to_screen(x: float, y: float, flip: bool = False) -> tuple[float, float]:
    scaled_x = x * SCREEN_W / STANDARD_PITCH_WIDTH
    scaled_y = y * SCREEN_H / STANDARD_PITCH_HEIGHT
    return (
        (scaled_x, scaled_y) if not flip else (SCREEN_W - scaled_x, SCREEN_H - scaled_y)
    )
