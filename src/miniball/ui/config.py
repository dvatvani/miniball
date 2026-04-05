"""Screen-space rendering constants: window size, pixel geometry, and colours.

These are only consumed by the game UI layer (``miniball.ui.game``) and the
screen-space coordinate transforms (``miniball.ui.coords``).  Simulation,
AI, and analytics code should import from ``miniball.config`` instead.
"""

from miniball.config import (
    BALL_RADIUS,
    PLAYER_RADIUS,
    STANDARD_GOAL_DEPTH,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)

# ── Window ──────────────────────────────────────────────────────────────────
SCREEN_W = 1200
SCREEN_H = 800
TITLE = "Miniball – 5-a-side Football"

# ── Pitch geometry in screen coordinates ───────────────────────────────
PITCH_L = 100
PITCH_R = 1100
PITCH_B = 75
PITCH_T = 725
PITCH_CX = (PITCH_L + PITCH_R) / 2
PITCH_CY = (PITCH_B + PITCH_T) / 2
GOAL_H = (PITCH_T - PITCH_B) * (STANDARD_GOAL_HEIGHT / STANDARD_PITCH_HEIGHT)
GOAL_DEPTH = (PITCH_R - PITCH_L) * (STANDARD_GOAL_DEPTH / STANDARD_PITCH_WIDTH)

# ── Screen-space rendering sizes ──────────────────────────────────────────────
SCREEN_PLAYER_RADIUS = (PITCH_R - PITCH_L) * (PLAYER_RADIUS / STANDARD_PITCH_WIDTH)
SCREEN_BALL_RADIUS = (PITCH_R - PITCH_L) * (BALL_RADIUS / STANDARD_PITCH_WIDTH)

# ── Input tuning ──────────────────────────────────────────────────────────────
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
C_COOLDOWN_RING = (255, 160, 50)  # amber  – radial cooldown arc
C_TEAM_A = (210, 40, 40)  # red   – left side
C_TEAM_B = (30, 100, 200)  # blue  – right side
C_HUD = (255, 255, 255)
C_HINT = (180, 180, 180)
