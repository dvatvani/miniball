import marimo

__generated_with = "0.22.0"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH
    from miniball.ai.utils import norm

    return STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH, norm, plt


@app.cell
def _():
    from glob import glob

    return (glob,)


@app.cell
def _(glob):
    match_files = sorted(glob("match_data/*.parquet"))
    return (match_files,)


@app.cell
def _(match_files, mo):
    selected_match = mo.ui.dropdown(
        match_files, value=match_files[-1], label="Select match"
    )
    selected_match
    return (selected_match,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspect frames
    """)
    return


@app.cell
def _(selected_match):
    from miniball.match_simulation import load_match

    match_result, df, frames = load_match(selected_match.value)
    return frames, match_result


@app.cell
def _(match_result, mo):
    team_selector = mo.ui.radio(
        [match_result.home_team, match_result.away_team],
        value=match_result.home_team,
    )
    team_selector
    return (team_selector,)


@app.cell
def _(mo):
    show_interception_point_selector = mo.ui.checkbox(
        False, label="Show projected interception point"
    )
    show_direction_selector = mo.ui.checkbox(False, label="Show intended direction")
    normalise_direction_selector = mo.ui.checkbox(False, label="Normalise direction")
    mo.hstack(
        [
            show_interception_point_selector,
            show_direction_selector,
            normalise_direction_selector,
        ]
    )
    return (
        normalise_direction_selector,
        show_direction_selector,
        show_interception_point_selector,
    )


@app.cell
def _(frames, mo):
    from wigglystuff import PlaySlider

    slider = mo.ui.anywidget(
        PlaySlider(
            min_value=0,
            max_value=len(frames) - 1,
            step=1,
            interval_ms=50,
            loop=False,
        )
    )
    slider
    return (slider,)


@app.cell
def _(frames, mo, slider):
    frame_selector = mo.ui.number(
        start=0, stop=len(frames) - 1, value=slider.value["value"], label="Frame"
    )
    frame_selector
    return (frame_selector,)


@app.cell
def _(frame_selector, frames, match_result, team_selector):
    frame = frames[int(frame_selector.value)]

    if team_selector.value == match_result.home_team:
        # state_a is the home-team perspective: all locations are in the global frame.
        state = frame.state_a
        actions = frame.actions_a
    else:
        state = frame.state_b
        actions = frame.actions_b
    return actions, frame, state


@app.cell
def _(
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    actions,
    frame,
    norm,
    normalise_direction_selector,
    plt,
    show_direction_selector,
    show_interception_point_selector,
    state,
):
    def plot_frame(
        state,
        actions=None,
        show_action_directions=False,
        normalise_action_directions=False,
        show_projected_ball_interception_points=False,
    ):
        _fig, _ax = plt.subplots(figsize=(8, 6))
        _ax.set_xlim(0, STANDARD_PITCH_WIDTH)
        _ax.set_ylim(0, STANDARD_PITCH_HEIGHT)
        _ax.set_xlabel("(attacking direction →)")
        _ax.set_ylabel("")

        _null_direction = (0.0, 0.0)

        _ball_interception_times = state.ball.interception_times(state.all_players)

        for _player in state.all_players:
            _ax.scatter(
                [_player.location[0]],
                [_player.location[1]],
                s=120,
                facecolor="w",
                edgecolor="b" if _player.is_teammate else "r",
                alpha=0.9 if _player.cooldown_timer == 0 else 0.2,
                linewidths=0.5,
                zorder=5,
            )
            _ax.annotate(
                str(_player.number),
                _player.location,
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                weight="bold",
                color="b" if _player.is_teammate else "r",
            )

            if _player.is_teammate:
                _action = actions.get(_player.number) if actions is not None else None
                direction = _action["direction"] if _action else _null_direction
                if normalise_action_directions:
                    direction = norm(*direction)

                if direction[0] != 0.0 or direction[1] != 0.0:
                    if show_action_directions:
                        _ax.arrow(
                            *_player.location,
                            *direction,
                            head_width=0.9,
                            head_length=1.0,
                            length_includes_head=True,
                            color="b" if _player.is_teammate else "r",
                            alpha=0.8,
                            linewidth=1,
                            zorder=6,
                        )
            if show_projected_ball_interception_points:
                _interception_location = _player.intercept_point(state.ball)
                _ax.plot(
                    [_interception_location[0], _player.location[0]],
                    [_interception_location[1], _player.location[1]],
                    c="lightblue" if _player.is_teammate else "pink",
                    lw=0.5,
                    zorder=1,
                )

        # Ball
        _ax.scatter(
            *state.ball.location,
            s=50,
            color="gold",
            edgecolors="k",
            marker="o",
            zorder=5,
        )

        ball_path = state.ball.trace_path()
        if len(ball_path) > 1:
            for _i, (_start, _end) in enumerate(zip(ball_path[:-1], ball_path[1:])):
                _ax.arrow(
                    _start[0][0],
                    _start[0][1],
                    _end[0][0] - _start[0][0],
                    _end[0][1] - _start[0][1],
                    head_width=0.0 if _i < len(ball_path) - 2 else 0.9,
                    head_length=0.0 if _i < len(ball_path) - 2 else 1.0,
                    length_includes_head=True,
                    color="k",
                    alpha=0.8,
                    linewidth=1,
                    zorder=6,
                )

        _time = state.match_state.match_time_seconds
        _score_a = state.match_state.team_current_score
        _score_b = state.match_state.opposition_current_score
        _ax.set_title(
            f"Frame {frame.frame_number} — {_time:.2f}s  "
            f"({state.match_state.team_current_score}–{state.match_state.opposition_current_score})"
        )
        _ax.grid(True, alpha=0.3)
        return plt.gca()

    plot_frame(
        state,
        actions,
        show_action_directions=show_direction_selector.value,
        normalise_action_directions=normalise_direction_selector.value,
        show_projected_ball_interception_points=show_interception_point_selector.value,
    )
    return


@app.cell
def _(state):
    state.all_players
    return


@app.cell
def _(actions):
    actions
    return


if __name__ == "__main__":
    app.run()
