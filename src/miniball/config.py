# ── Window ──────────────────────────────────────────────────────────────────
SCREEN_W = 1200
SCREEN_H = 800
TITLE = "Miniball – 5-a-side Football"


# ── Pitch coordinate system ──────────────────────────────────────────────────
# Origin is the defensive right corner of the pitch.
STANDARD_PITCH_WIDTH = 120
STANDARD_PITCH_HEIGHT = 80

# Goal dimensions in normalised "global" coordinates
STANDARD_GOAL_HEIGHT = 12  # vertical opening of each goal
STANDARD_GOAL_DEPTH = 4  # how far the goal box extends behind the goal line


# Game settings, in normalised "global" coordinates
PLAYER_RADIUS = 2.0
BALL_RADIUS = 1.0
PLAYER_SPEED = 12  # normalised units / s
BALL_DRAG = (
    0.07  # speed loss, normalised units / s per second (free ball, linear model)
)
STRIKE_SPEED = 60  # normalised units / s on a strike
GAME_DURATION = 120.0  # seconds per half (full game = one period)
TACKLE_COOLDOWN = 1.0  # seconds unable to gain the ball after being tackled
STRIKE_COOLDOWN = 1.0  # seconds before the kicker can engage with the ball again
COOLDOWN_ALPHA = 90  # draw opacity (0–255) while on cooldown (~35 %)
JOY_DEAD_ZONE = 0.15  # ignore analogue stick values below this magnitude
JOY_SWITCH_THRESHOLD = 0.7  # right-stick magnitude that triggers a player switch

# ── Pitch geometry in screen coordinates ───────────────────────────────
PITCH_L = 100
PITCH_R = 1100
PITCH_B = 75
PITCH_T = 725
PITCH_CX = (PITCH_L + PITCH_R) / 2
PITCH_CY = (PITCH_B + PITCH_T) / 2
GOAL_H = (PITCH_T - PITCH_B) * (
    STANDARD_GOAL_HEIGHT / STANDARD_PITCH_HEIGHT
)  # vertical opening of each goal
GOAL_DEPTH = (PITCH_R - PITCH_L) * (
    STANDARD_GOAL_DEPTH / STANDARD_PITCH_WIDTH
)  # how far the goal box extends behind the goal line

# ── Physics / timings ────────────────────────────────────────────────────────
GAME_ENGINE_PLAYER_RADIUS = (PITCH_R - PITCH_L) * (PLAYER_RADIUS / STANDARD_PITCH_WIDTH)
GAME_ENGINE_BALL_RADIUS = (PITCH_R - PITCH_L) * (BALL_RADIUS / STANDARD_PITCH_WIDTH)
GAME_ENGINE_PLAYER_SPEED = (PITCH_R - PITCH_L) * (PLAYER_SPEED / STANDARD_PITCH_WIDTH)
GAME_ENGINE_BALL_DRAG = (PITCH_R - PITCH_L) * (BALL_DRAG / STANDARD_PITCH_WIDTH)
GAME_ENGINE_STRIKE_SPEED = (PITCH_R - PITCH_L) * (STRIKE_SPEED / STANDARD_PITCH_WIDTH)

# ── Colours ──────────────────────────────────────────────────────────────────
C_GRASS = (34, 139, 34)
C_LINE = (255, 255, 255)
C_GOAL = (220, 220, 220)
C_BALL = (255, 255, 255)
C_BALL_OUTLINE = (20, 20, 20)
C_PLAYER_OUTLINE = (20, 20, 20)
C_CONTROLLED = (255, 215, 0)  # yellow  – keyboard-controlled player
C_POSSESSION = (255, 255, 255)  # white   – player who has the ball
C_COOLDOWN_RING = (255, 160, 50)  # amber  – radial cooldown arc
C_TEAM_A = (210, 40, 40)  # red   – left side
C_TEAM_B = (30, 100, 200)  # blue  – right side
C_HUD = (255, 255, 255)
C_HINT = (180, 180, 180)
