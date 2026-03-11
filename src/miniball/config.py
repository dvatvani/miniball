# ── Window ──────────────────────────────────────────────────────────────────
SCREEN_W = 1200
SCREEN_H = 800
TITLE = "Miniball – 5-a-side Football"

# ── Pitch geometry in game engine coordinates ───────────────────────────────
PITCH_L = 100
PITCH_R = 1100
PITCH_B = 75
PITCH_T = 725
PITCH_CX = (PITCH_L + PITCH_R) / 2
PITCH_CY = (PITCH_B + PITCH_T) / 2
GOAL_H = 140  # vertical opening of each goal
GOAL_DEPTH = 32  # how far the goal box extends behind the goal line

# ── Physics / timings ────────────────────────────────────────────────────────
PLAYER_RADIUS = 18
BALL_RADIUS = 10
PLAYER_SPEED = 120  # px / s
BALL_DRAG = 0.7  # speed loss per second (free ball, linear model)
STRIKE_SPEED = 550  # px / s on a strike
MAX_BALL_SPEED = 700
GAME_DURATION = 60.0  # seconds per half (full game = one period)
TACKLE_COOLDOWN = 1.0  # seconds unable to gain the ball after being tackled
STRIKE_COOLDOWN = 1.0  # seconds before the kicker can re-absorb their own strike
COOLDOWN_ALPHA = 90  # draw opacity (0–255) while on cooldown (~35 %)
JOY_DEAD_ZONE = 0.15  # ignore analogue stick values below this magnitude
JOY_SWITCH_THRESHOLD = 0.7  # right-stick magnitude that triggers a player switch

# ── Colours ──────────────────────────────────────────────────────────────────
C_GRASS = (34, 139, 34)
C_LINE = (255, 255, 255)
C_GOAL = (220, 220, 220)
C_BALL = (255, 255, 255)
C_BALL_OUTLINE = (20, 20, 20)
C_PLAYER_OUTLINE = (20, 20, 20)
C_CONTROLLED = (255, 215, 0)  # yellow  – keyboard-controlled player
C_POSSESSION = (255, 255, 255)  # white   – player who has the ball
C_TEAM_A = (210, 40, 40)  # red   – left side
C_TEAM_B = (30, 100, 200)  # blue  – right side
C_HUD = (255, 255, 255)
C_HINT = (180, 180, 180)

# ── Pitch coordinate system ──────────────────────────────────────────────────
# Origin is the defensive right corner of the pitch.
STANDARD_PITCH_WIDTH = 120
STANDARD_PITCH_HEIGHT = 80
