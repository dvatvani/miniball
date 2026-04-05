import matplotlib.pyplot as plt
import numpy as np

from miniball.config import (
    STANDARD_GOAL_DEPTH,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
)
from miniball.geometry import norm

TEAM_HALF_PITCH_COORDS = np.array(
    [
        [0, 0],
        [STANDARD_PITCH_WIDTH / 2, 0],
        [STANDARD_PITCH_WIDTH / 2, STANDARD_PITCH_HEIGHT],
        [0, STANDARD_PITCH_HEIGHT],
    ]
)

OPPOSITION_HALF_PITCH_COORDS = np.array(
    [
        [STANDARD_PITCH_WIDTH / 2, 0],
        [STANDARD_PITCH_WIDTH, 0],
        [STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT],
        [STANDARD_PITCH_WIDTH / 2, STANDARD_PITCH_HEIGHT],
    ]
)

TEAM_GOAL_COORDS = np.array(
    [
        [0, STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2],
        [
            0 - STANDARD_GOAL_DEPTH,
            STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2,
        ],
        [
            0 - STANDARD_GOAL_DEPTH,
            STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2,
        ],
        [0, STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2],
    ]
)

OPPOSITION_GOAL_COORDS = np.array(
    [
        [STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2],
        [
            STANDARD_PITCH_WIDTH + STANDARD_GOAL_DEPTH,
            STANDARD_PITCH_HEIGHT / 2 - STANDARD_GOAL_HEIGHT / 2,
        ],
        [
            STANDARD_PITCH_WIDTH + STANDARD_GOAL_DEPTH,
            STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2,
        ],
        [STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT / 2 + STANDARD_GOAL_HEIGHT / 2],
    ]
)


def plot_state(
    state,
    actions=None,
    show_projected_ball_path=True,
    show_action_directions=False,
    normalise_action_directions=False,
    show_projected_ball_interception_points=False,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(TEAM_GOAL_COORDS[:, 0].min(), OPPOSITION_GOAL_COORDS[:, 0].max())
    ax.set_ylim(
        TEAM_HALF_PITCH_COORDS[:, 1].min(), OPPOSITION_HALF_PITCH_COORDS[:, 1].max()
    )
    ax.set_xlabel("attacking direction →")
    ax.set_ylabel("")
    ax.fill(
        TEAM_HALF_PITCH_COORDS[:, 0],
        TEAM_HALF_PITCH_COORDS[:, 1],
        color="blue",
        alpha=0.3,
        zorder=1,
    )
    ax.fill(
        OPPOSITION_HALF_PITCH_COORDS[:, 0],
        OPPOSITION_HALF_PITCH_COORDS[:, 1],
        color="red",
        alpha=0.3,
        zorder=1,
    )
    ax.fill(
        TEAM_GOAL_COORDS[:, 0],
        TEAM_GOAL_COORDS[:, 1],
        color="blue",
        alpha=0.3,
        zorder=1,
    )
    ax.fill(
        OPPOSITION_GOAL_COORDS[:, 0],
        OPPOSITION_GOAL_COORDS[:, 1],
        color="red",
        alpha=0.3,
        zorder=1,
    )
    null_direction = (0.0, 0.0)
    for player in state.all_players:
        ax.scatter(
            player.location[0],
            player.location[1],
            s=120,
            facecolor="w",
            edgecolor="blue" if player.is_teammate else "red",
            alpha=0.9 if player.cooldown_timer == 0 else 0.2,
            linewidths=2.0,
            zorder=5,
        )
        ax.annotate(
            str(player.number),
            player.location,
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            weight="bold",
            color="b" if player.is_teammate else "r",
        )

        if player.is_teammate:
            _action = actions.get(player.number) if actions is not None else None
            direction = _action["direction"] if _action else null_direction
            if normalise_action_directions:
                direction = norm(*direction)

            if direction[0] != 0.0 or direction[1] != 0.0:
                if show_action_directions:
                    ax.arrow(
                        *player.location,
                        *direction,
                        head_width=0.9,
                        head_length=1.0,
                        length_includes_head=True,
                        color="b" if player.is_teammate else "r",
                        alpha=0.8,
                        linewidth=1,
                        zorder=6,
                    )
        if show_projected_ball_interception_points:
            interception_location = player.intercept_point(state.ball)
            ax.plot(
                [interception_location[0], player.location[0]],
                [interception_location[1], player.location[1]],
                c="blue" if player.is_teammate else "red",
                ls="--",
                lw=0.5,
                zorder=2,
            )

    # Ball
    ax.scatter(
        *state.ball.location,
        s=50,
        color="gold",
        edgecolors="k",
        marker="o",
        zorder=5,
    )

    if show_projected_ball_path:
        ball_path = state.ball.trace_path()
        if len(ball_path) > 1:
            for i, (start, end) in enumerate(zip(ball_path[:-1], ball_path[1:])):
                ax.arrow(
                    start[0][0],
                    start[0][1],
                    end[0][0] - start[0][0],
                    end[0][1] - start[0][1],
                    head_width=0.0 if i < len(ball_path) - 2 else 0.9,
                    head_length=0.0 if i < len(ball_path) - 2 else 1.0,
                    length_includes_head=True,
                    color="k",
                    alpha=0.8,
                    linewidth=1,
                    zorder=6,
                )

    time = state.match_state.match_time_seconds
    score_a = state.match_state.team_current_score
    score_b = state.match_state.opposition_current_score
    ax.set_title(f"{time:.2f}s  ({score_a}–{score_b})")
    ax.grid(False)
    ax.axis("off")
    return plt.gca()
