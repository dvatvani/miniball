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

    return pl, plt


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
        match_files, value=match_files[0], label="Select match"
    )
    selected_match
    return (selected_match,)


@app.cell
def _(pl, selected_match):
    match_data = pl.read_parquet(selected_match.value)
    match_data
    return (match_data,)


@app.cell
def _(match_data, pl, plt):
    avg_locations = (
        match_data.group_by(["team", "player_number"])
        .agg(
            [
                pl.col("pos_x_global").mean().alias("x_mean"),
                pl.col("pos_y_global").mean().alias("y_mean"),
                pl.len().alias("n"),
            ]
        )
        .sort(["team", "player_number"])
    )

    avg_pd = avg_locations.to_pandas()

    teams = avg_pd["team"].unique()
    color_map = {team: plt.cm.tab10(i % 10) for i, team in enumerate(teams)}

    fig_avg, ax_avg = plt.subplots(figsize=(10, 7))
    ax_avg.set_xlim(0, 120)
    ax_avg.set_ylim(0, 80)
    ax_avg.set_xlabel("Global X")
    ax_avg.set_ylabel("Global Y")
    ax_avg.set_title("Average Global Locations of Players by Team")
    ax_avg.grid(True, alpha=0.3)

    for team in teams:
        subset = avg_pd[avg_pd["team"] == team]
        ax_avg.scatter(
            subset["x_mean"],
            subset["y_mean"],
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
                (row["x_mean"], row["y_mean"]),
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
def _(match_data, mo):
    frame_selector = mo.ui.number(
        start=0, stop=match_data.get_column("frame_number").max(), value=0
    )
    frame_selector
    return (frame_selector,)


@app.cell
def _(frame_selector, match_data, pl):
    _frame_df = match_data.filter(
        pl.col("frame_number") == frame_selector.value
    ).to_pandas()
    _frame_df
    return


@app.cell
def _(color_map, frame_selector, match_data, pl, plt):
    _frame_df = match_data.filter(
        pl.col("frame_number") == frame_selector.value
    ).to_pandas()
    _fig_frame, _ax_frame = plt.subplots(figsize=(8, 6))
    _ax_frame.set_xlim(0, 120)
    _ax_frame.set_ylim(0, 80)
    _ax_frame.set_xlabel("Global X")
    _ax_frame.set_ylabel("Global Y")

    for _team in _frame_df["team"].unique():
        _sub = _frame_df[_frame_df["team"] == _team]
        _ax_frame.scatter(
            _sub["pos_x_global"],
            _sub["pos_y_global"],
            s=120,
            facecolor="w",
            edgecolor=color_map.get(_team, plt.cm.tab10(0)),
            alpha=0.9,
            label=_team,
            linewidths=0.5,
        )
        for _, _row in _sub.iterrows():
            _ax_frame.annotate(
                int(_row["player_number"]),
                (_row["pos_x_global"], _row["pos_y_global"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                weight="bold",
                color="black",
            )

            # draw movement arrow if action deltas are present and non-zero
            try:
                _dx = float(_row["action_dx_global"])
                _dy = float(_row["action_dy_global"])
            except Exception:
                _dx = 0.0
                _dy = 0.0

            # filter out NaN (NaN != NaN) and zero-length vectors
            if not (_dx != _dx or _dy != _dy) and not (_dx == 0.0 and _dy == 0.0):
                _ax_frame.arrow(
                    _row["pos_x_global"],
                    _row["pos_y_global"],
                    _dx,
                    _dy,
                    head_width=0.9,
                    head_length=1.0,
                    length_includes_head=True,
                    color=color_map.get(_team, plt.cm.tab10(0)),
                    alpha=0.8,
                    linewidth=1,
                    zorder=6,
                )

    # Plot ball (use first row's ball position for the frame)
    _ball_x = _frame_df["ball_x"].iloc[0]
    _ball_y = _frame_df["ball_y"].iloc[0]
    _ax_frame.scatter(
        _ball_x,
        _ball_y,
        s=50,
        color="gold",
        edgecolors="k",
        marker="o",
        label="ball",
        zorder=5,
    )

    # draw movement arrow if action deltas are present and non-zero
    try:
        _dx = float(_frame_df["ball_vx_global"].iloc[0])
        _dy = float(_frame_df["ball_vy_global"].iloc[0])
    except Exception:
        _dx = 0.0
        _dy = 0.0

    # filter out NaN (NaN != NaN) and zero-length vectors
    if not (_dx != _dx or _dy != _dy) and not (_dx == 0.0 and _dy == 0.0):
        print("test")
        _ax_frame.arrow(
            _frame_df.iloc[0]["ball_x_global"],
            _frame_df.iloc[0]["ball_y_global"],
            _dx,
            _dy,
            head_width=0.9,
            head_length=1.0,
            length_includes_head=True,
            color="k",
            alpha=0.8,
            linewidth=1,
            zorder=6,
        )

    _time = _frame_df["match_time_seconds"].iloc[0]
    _ax_frame.set_title(f"Frame {frame_selector.value} — time {_time:.2f}s")
    _ax_frame.legend(title="Team", loc="upper right", bbox_to_anchor=(1.15, 1))

    _ax_frame.grid(True, alpha=0.3)
    plt.gca()
    return


@app.cell
def _():
    # multi_widget
    return


if __name__ == "__main__":
    app.run()
