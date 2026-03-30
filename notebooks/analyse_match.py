import marimo

__generated_with = "0.19.9"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl
    from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH

    return STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH, pl, plt


@app.cell
def _():
    from glob import glob

    return (glob,)


@app.cell
def _(glob):
    match_files = glob("match_data/*.parquet")
    return (match_files,)


@app.cell
def _(match_files, mo):
    selected_match = mo.ui.dropdown(
        match_files, value=match_files[-1], label="Select match"
    )
    selected_match
    return (selected_match,)


@app.cell
def _(pl, selected_match):
    match_data = pl.read_parquet(selected_match.value)
    match_data
    return (match_data,)


@app.cell
def _(STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH, match_data, pl, plt):
    # Average positions in team-relative coordinates (team always attacks right).
    # Using team coords means the distribution is comparable across matches
    # regardless of which physical side each team occupied.
    avg_locations = (
        match_data.group_by(["team", "is_home", "player_number"])
        .agg(
            [
                pl.col("pos_x").mean().alias("x_mean"),
                pl.col("pos_y").mean().alias("y_mean"),
                pl.len().alias("n"),
            ]
        )
        .sort(["team", "player_number"])
    )

    avg_pd = avg_locations.to_pandas()

    team_is_home = avg_pd.groupby("team").is_home.first().to_dict()
    teams = team_is_home.keys()

    color_map = {team: plt.cm.tab10(i % 10) for i, team in enumerate(teams)}

    fig_avg, ax_avg = plt.subplots(figsize=(10, 7))
    ax_avg.set_xlim(0, STANDARD_PITCH_WIDTH)
    ax_avg.set_ylim(0, STANDARD_PITCH_HEIGHT)
    ax_avg.set_xlabel("Team X (attacking →)")
    ax_avg.set_ylabel("Team Y")
    ax_avg.set_title("Average Positions by Team (team-relative coordinates)")
    ax_avg.grid(True, alpha=0.3)

    for team in teams:
        subset = avg_pd[avg_pd["team"] == team]
        ax_avg.scatter(
            subset["x_mean"]
            if team_is_home[team]
            else STANDARD_PITCH_WIDTH - subset["x_mean"],
            subset["y_mean"]
            if team_is_home[team]
            else STANDARD_PITCH_HEIGHT - subset["y_mean"],
            s=(50 + subset["n"] / subset["n"].max() * 150),
            color=color_map[team],
            alpha=0.9,
            label=team,
            edgecolors="k",
            linewidths=0.5,
        )
        for _, row in subset.iterrows():
            ax_avg.annotate(
                int(row["player_number"]),
                (
                    row["x_mean"]
                    if team_is_home[team]
                    else STANDARD_PITCH_WIDTH - row["x_mean"],
                    row["y_mean"]
                    if team_is_home[team]
                    else STANDARD_PITCH_HEIGHT - row["y_mean"],
                ),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                weight="bold",
                color="black",
            )

    ax_avg.legend(title="Team", loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.gca()
    return (color_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspect frames
    """)
    return


@app.cell
def _(selected_match):
    from miniball.match_simulation import reconstruct_frames

    frames = reconstruct_frames(selected_match.value)
    return (frames,)


@app.cell
def _(frames, mo):
    frame_selector = mo.ui.number(start=0, stop=len(frames) - 1, value=0)
    frame_selector
    return (frame_selector,)


@app.cell
def _(
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    color_map,
    frame_selector,
    frames,
    match_data,
    plt,
):
    _PITCH_W = STANDARD_PITCH_WIDTH
    _PITCH_H = STANDARD_PITCH_HEIGHT

    # Build a team-name lookup so we can map is_home → color
    _teams_df = match_data.select(["team", "is_home"]).unique()
    _is_home_to_team = dict(
        zip(_teams_df["is_home"].to_list(), _teams_df["team"].to_list())
    )

    _frame = frames[frame_selector.value]
    # state_a is the home-team perspective: all locations are in the global frame.
    _state = _frame.state_a

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.set_xlim(0, _PITCH_W)
    _ax.set_ylim(0, _PITCH_H)
    _ax.set_xlabel("Global X (home attacks →)")
    _ax.set_ylabel("Global Y")

    _null_direction = (0.0, 0.0)

    for _player in _state.all_players:
        _gx, _gy = _player.global_location
        _team_name = _is_home_to_team.get(_player.is_home, "?")
        _color = color_map.get(_team_name, plt.cm.tab10(0))

        _ax.scatter(
            _gx,
            _gy,
            s=120,
            facecolor="w",
            edgecolor=_color,
            alpha=0.9,
            linewidths=0.5,
        )
        _ax.annotate(
            str(_player.number),
            (_gx, _gy),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            weight="bold",
            color="black",
        )

        # Action arrow: actions are stored in each team's own frame, so team B's
        # actions need to be negated to convert them to the global frame.
        if _player.is_home:
            _action = _frame.actions_a.get(_player.number)
            _dx, _dy = _action["direction"] if _action else _null_direction
        else:
            _action = _frame.actions_b.get(_player.number)
            if _action:
                _dx, _dy = -_action["direction"][0], -_action["direction"][1]
            else:
                _dx, _dy = _null_direction

        if _dx != 0.0 or _dy != 0.0:
            _ax.arrow(
                _gx,
                _gy,
                _dx,
                _dy,
                head_width=0.9,
                head_length=1.0,
                length_includes_head=True,
                color=_color,
                alpha=0.8,
                linewidth=1,
                zorder=6,
            )

    # Add a legend entry per team (one scatter call per team for the label)
    for _is_home, _tname in _is_home_to_team.items():
        _ax.scatter(
            [],
            [],
            s=80,
            facecolor="w",
            edgecolor=color_map.get(_tname, plt.cm.tab10(0)),
            linewidths=1.5,
            label=_tname,
        )

    # Ball
    _bx, _by = _state.global_ball_location
    _bvx, _bvy = _state.global_ball_velocity
    _ax.scatter(
        _bx,
        _by,
        s=50,
        color="gold",
        edgecolors="k",
        marker="o",
        label="ball",
        zorder=5,
    )
    if _bvx != 0.0 or _bvy != 0.0:
        _ax.arrow(
            _bx,
            _by,
            _bvx,
            _bvy,
            head_width=0.9,
            head_length=1.0,
            length_includes_head=True,
            color="k",
            alpha=0.8,
            linewidth=1,
            zorder=6,
        )

    _time = _state.match_state.match_time_seconds
    _score_a = _state.match_state.team_current_score
    _score_b = _state.match_state.opposition_current_score
    _ax.set_title(
        f"Frame {_frame.frame_number} — {_time:.2f}s  "
        f"({_is_home_to_team.get(True, 'Home')} {_score_a}–{_score_b} "
        f"{_is_home_to_team.get(False, 'Away')})"
    )
    _ax.legend(title="Team", loc="upper right", bbox_to_anchor=(1.15, 1))
    _ax.grid(True, alpha=0.3)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
