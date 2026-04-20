# ── Pitch coordinate system ──────────────────────────────────────────────────
# Origin is the defensive right corner of the pitch.
STANDARD_PITCH_WIDTH = 120
STANDARD_PITCH_HEIGHT = 80

# Goal dimensions in normalised "global" coordinates
STANDARD_GOAL_HEIGHT = 18  # vertical opening of each goal
STANDARD_GOAL_DEPTH = 4  # how far the goal box extends behind the goal line


# Game settings, in standard pitch coordinates (120x80 pitch)
PLAYER_RADIUS = 2.0
BALL_RADIUS = 1.0
PLAYER_SPEED = 12  # normalised units / s
BALL_DECEL = (
    10.5  # deceleration (units/s²): speed decreases by BALL_DECEL · dt each frame
)
MAX_STRIKE_SPEED = 50  # normalised units / s on a full-weight (1.0) strike
GAME_DURATION = 120.0  # seconds per half (full game = one period)
TACKLE_COOLDOWN = 1.0  # seconds unable to gain the ball after being tackled
STRIKE_COOLDOWN = 1.0  # seconds before the kicker can engage with the ball again


def STRIKE_ANGULAR_ERROR_DEGREES_FN(weight: float) -> float:
    """Max angular error in degrees for a strike at the given weight (linear: 0 → 4)."""
    return weight**3 * 6.0


# Strike weights for each PS4 button (configurable)
STRIKE_WEIGHT_CROSS = 0.75  # Cross button — balanced
STRIKE_WEIGHT_SQUARE = 1.0  # Square button — full power
STRIKE_WEIGHT_TRIANGLE = 0.25  # Triangle button — tap
STRIKE_WEIGHT_CIRCLE = 0.50  # Circle button — light
