# ── Pitch coordinate system ──────────────────────────────────────────────────
# Origin is the defensive right corner of the pitch.
STANDARD_PITCH_WIDTH = 120
STANDARD_PITCH_HEIGHT = 80

# Goal dimensions in normalised "global" coordinates
STANDARD_GOAL_HEIGHT = 18  # vertical opening of each goal
STANDARD_GOAL_DEPTH = 4  # how far the goal box extends behind the goal line


# Game settings, in normalised "global" coordinates
PLAYER_RADIUS = 2.0
BALL_RADIUS = 1.0
PLAYER_SPEED = 12  # normalised units / s
BALL_DRAG = 7 / 12  # drag coefficient (1/s): v *= (1 − BALL_DRAG · dt) each frame
STRIKE_SPEED = 60  # normalised units / s on a strike
GAME_DURATION = 120.0  # seconds per half (full game = one period)
TACKLE_COOLDOWN = 1.0  # seconds unable to gain the ball after being tackled
STRIKE_COOLDOWN = 1.0  # seconds before the kicker can engage with the ball again
