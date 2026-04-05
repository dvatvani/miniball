import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from miniball.ai.utils.vis import plot_state

    return (plot_state,)


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
    from miniball.simulation import load_match

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
    show_ball_path_selector = mo.ui.checkbox(True, label="Show projected ball path")
    show_interception_point_selector = mo.ui.checkbox(
        False, label="Show projected interception point"
    )
    show_direction_selector = mo.ui.checkbox(
        False, label="Show player intended direction"
    )
    normalise_direction_selector = mo.ui.checkbox(
        False, label="Normalise player direction"
    )
    mo.hstack(
        [
            show_ball_path_selector,
            show_interception_point_selector,
            show_direction_selector,
            normalise_direction_selector,
        ]
    )
    return (
        normalise_direction_selector,
        show_ball_path_selector,
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
    return actions, state


@app.cell
def _(
    actions,
    normalise_direction_selector,
    plot_state,
    show_ball_path_selector,
    show_direction_selector,
    show_interception_point_selector,
    state,
):
    plot_state(
        state,
        actions,
        show_projected_ball_path=show_ball_path_selector.value,
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
