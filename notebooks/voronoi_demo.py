import marimo

__generated_with = "0.19.9"
app = marimo.App(width="wide")


@app.cell
def _():
    from miniball.ai.utils.geometry import (
        players_bounded_voronoi,
        plot_bounded_voronoi,
    )
    from miniball.teams import teams_list

    return players_bounded_voronoi, plot_bounded_voronoi, teams_list


@app.cell
def _(teams_list):
    from miniball.ai import PlayerState

    # The list of player states is available during the simulation runtime, so we don't have to construct them in this way.
    # This construction is to recreate the objects as they'd be available at simulation runtime

    players = [
        PlayerState(
            number=p.number,
            is_teammate=True,
            has_ball=False,
            location=(p.x, p.y),
            cooldown_timer=0.0,
        )
        for p in teams_list[0].players
    ]
    return (players,)


@app.cell
def _(players, players_bounded_voronoi):
    vor = players_bounded_voronoi(players)
    vor.points, vor.region_centroids, vor.regions
    return (vor,)


@app.cell
def _(plot_bounded_voronoi, vor):
    plot_bounded_voronoi(vor)
    return


if __name__ == "__main__":
    app.run()
