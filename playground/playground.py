import marimo

__generated_with = "0.19.9"
app = marimo.App(width="wide")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    from miniball.coordinate_transformations import team_to_global

    def away(tx: float, ty: float) -> dict[str, float]:
        gx, gy = team_to_global(tx, ty, is_home=False)
        return {"x": gx, "y": gy}

    players = [
        dict(name="GK", number=1, x=10, y=40, is_home=True),
        dict(name="Defender 1", number=2, x=50, y=60, is_home=True),
        dict(name="Defender 2", number=3, x=50, y=20, is_home=True),
        dict(name="Forward 1", number=4, x=100, y=50, is_home=True),
        dict(name="Forward 2", number=5, x=100, y=30, is_home=True),
        dict(name="GK", number=1, **away(10, 40), is_home=False),
        dict(name="Defender 1", number=2, **away(50, 60), is_home=False),
        dict(name="Defender 2", number=3, **away(50, 20), is_home=False),
        dict(name="Forward 1", number=4, **away(100, 50), is_home=False),
        dict(name="Forward 2", number=5, **away(100, 30), is_home=False),
    ]
    return (players,)


@app.cell
def _(dynamic_data_x, dynamic_data_y, plt):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 80)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Drag any puck - closest one will move")
    ax2.grid(True, alpha=0.3)

    def draw_with_crosshairs(ax, widget):
        x, y = widget.x[0], widget.y[0]
        ax.scatter(dynamic_data_x, dynamic_data_y, alpha=0.6)
        ax.axvline(x, color="red", linestyle="--", alpha=0.7)
        ax.axhline(y, color="red", linestyle="--", alpha=0.7)
        ax.set_title(f"Position: ({x:.2f}, {y:.2f})")
        ax.grid(True, alpha=0.3)

    # multi_puck = ChartPuck.from_callback(
    #     draw_fn=draw_with_crosshairs,
    #     fig2,
    #     x=[p['x'] for p in players],
    #     y=[p['y'] for p in players],
    #     puck_color=["#ff0000" if p['is_home'] else "#0000ff" for p in players],
    # )
    # plt.close(fig2)
    return


@app.cell
def _():
    # multi_widget = mo.ui.anywidget(multi_puck)
    return


@app.cell
def _():
    # multi_widget
    return


@app.cell(hide_code=True)
def _(np, players):
    import matplotlib.pyplot as pl

    import scipy as sp
    import scipy.spatial
    from types import SimpleNamespace

    _players = np.array(
        [
            [i["x"] for i in players],
            [i["y"] for i in players],
        ]
    ).T
    bounding_box = np.array([0.0, 120.0, 0.0, 80.0])  # [x_min, x_max, y_min, y_max]

    def bounded_voronoi(
        players,
        bounding_box=np.array([0.0, 120.0, 0.0, 80.0]),  # [x_min, x_max, y_min, y_max]
    ):
        # Mirror points
        points_center = players
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
        points = np.concat(
            [points_center, points_left, points_right, points_down, points_up]
        )
        # Compute Voronoi
        vor = sp.spatial.Voronoi(points)
        return SimpleNamespace(
            vertices=vor.vertices,
            filtered_points=points_center,
            filtered_regions=[
                vor.regions[i] for i in vor.point_region[: vor.npoints // 5]
            ],
        )

    def centroid_region(vertices):
        # Polygon's signed area
        A = 0
        # Centroid's x
        C_x = 0
        # Centroid's y
        C_y = 0
        for i in range(0, len(vertices) - 1):
            s = (
                vertices[i, 0] * vertices[i + 1, 1]
                - vertices[i + 1, 0] * vertices[i, 1]
            )
            A = A + s
            C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
            C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
        A = 0.5 * A
        C_x = (1.0 / (6.0 * A)) * C_x
        C_y = (1.0 / (6.0 * A)) * C_y
        return np.array([[C_x, C_y]])

    vor = bounded_voronoi(_players, bounding_box)

    def plot_bounded_voronoi(vor):
        fig = pl.figure()
        ax = fig.gca()
        # Plot initial points
        ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], "b.")
        # Plot ridges points
        for region in vor.filtered_regions:
            vertices = vor.vertices[region, :]
            ax.plot(vertices[:, 0], vertices[:, 1], "go")
        # Plot ridges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            ax.plot(vertices[:, 0], vertices[:, 1], "k-")
        # Compute and plot centroids
        centroids = []
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            centroid = centroid_region(vertices)
            centroids.append(list(centroid[0, :]))
            ax.plot(centroid[:, 0], centroid[:, 1], "r.")
        return fig

    plot_bounded_voronoi(vor)
    return


if __name__ == "__main__":
    app.run()
