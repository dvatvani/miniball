from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.path import Path

from miniball.ai.interface import PlayerState
from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH


def bounded_voronoi(
    points: np.ndarray,  # N x 2 array of points
    bounding_box=np.array(
        [0.0, STANDARD_PITCH_WIDTH, 0.0, STANDARD_PITCH_HEIGHT]
    ),  # [x_min, x_max, y_min, y_max]
):
    """Compute a bounded Voronoi diagram clipped to ``bounding_box``.

    Returns a ``SimpleNamespace`` with:

    ``points``
        The original N × 2 input points.
    ``regions``
        List of N arrays, each of shape (M, 2), containing the (x, y)
        coordinates of the polygon vertices for the corresponding input point's
        Voronoi cell.  Vertices are in the order returned by scipy; the polygon
        is open (last vertex does not repeat the first).
    """
    points_center = points
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.concatenate(
        [points_center, points_left, points_right, points_down, points_up]
    )
    vor = scipy.spatial.Voronoi(points)
    regions = [
        vor.vertices[vor.regions[i]] for i in vor.point_region[: vor.npoints // 5]
    ]
    region_centroids = [centroid_region(region) for region in regions]
    return SimpleNamespace(
        points=points_center, regions=regions, region_centroids=region_centroids
    )


def players_bounded_voronoi(players: list[PlayerState]):
    points = np.array([p["location"] for p in players], dtype=float)
    return bounded_voronoi(points)


def centroid_region(vertices):
    """Compute the centroid of a polygon."""
    # If the polygon is not closed, close it by appending the first vertex
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1]
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])


def location_in_polygon(location: Sequence[float], polygon: np.ndarray) -> bool:
    """Return ``True`` if ``location`` lies inside ``polygon``.

    Uses the winding-number algorithm via ``matplotlib.path.Path``.  The
    polygon does not need to be explicitly closed — the last edge back to the
    first vertex is added automatically.

    Parameters
    ----------
    location:
        ``(x, y)`` point to test.
    polygon:
        (M, 2) array of polygon vertex coordinates in order.
    """
    return bool(Path(polygon).contains_point((location[0], location[1])))


def player_in_polygon(player: PlayerState, polygon: np.ndarray) -> bool:
    """Return ``True`` if the player's location is inside ``polygon``."""
    return location_in_polygon(player["location"], polygon)


def players_in_polygon(
    players: list[PlayerState], polygon: np.ndarray
) -> list[PlayerState]:
    """Return the subset of ``players`` whose location is inside ``polygon``."""
    return [p for p in players if player_in_polygon(p, polygon)]


def plot_bounded_voronoi(vor: SimpleNamespace, plot_centroids: bool = True):
    fig = plt.figure()
    ax = fig.gca()
    # Plot input points
    ax.plot(vor.points[:, 0], vor.points[:, 1], "b.")
    # Plot region edges (close the polygon by appending the first vertex)
    for region in vor.regions:
        ax.fill(
            region[:, 0],
            region[:, 1],
            linewidth=1,
            fill=False,
        )
    if plot_centroids:
        for centroid in vor.region_centroids:
            ax.plot(centroid[:, 0], centroid[:, 1], "r.")
    return fig
