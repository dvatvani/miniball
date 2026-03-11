import numpy as np
from scipy.spatial import Delaunay

from miniball.ai.interface import BaseAI, GameState, PlayerState, TeamActions
from miniball.ai.utils import dist, goal_center, norm
from miniball.config import STANDARD_PITCH_HEIGHT, STANDARD_PITCH_WIDTH


class BaselineAI(BaseAI):
    """Zonal AI using Voronoi space creation and Delaunay-based passing.

    In possession
    ─────────────
    All players move toward the centroid of their Voronoi cell (computed from
    all ten players' current positions) to spread out and create space.

    The ball carrier's decision overrides the space-creation movement:

    1. Within ``STRIKE_RANGE`` of the attacking goal → face goal and strike.
    2. An opponent within ``PRESSURE_RANGE`` → pass to the furthest-forward
       teammate connected to the carrier by a Delaunay edge.  If no such
       teammate exists, strike toward goal instead.
    3. Otherwise → continue moving toward own Voronoi centroid.

    Out of possession
    ─────────────────
    1. The teammate nearest to the ball presses immediately.
    2. Every other player is assigned a *zone* defined by which formation
       position (from ``self.formation``) is closest to a given pitch location
       (i.e. the Voronoi diagram of the formation positions).
    3. If an opponent currently occupies a player's zone, that player moves to
       the point 90 % of the way along the line from the ball to the most
       forward (lowest x) opponent in the zone – cutting off the threat.
    4. If the zone is empty, the player returns to their formation position.
    """

    STRIKE_RANGE: float = 40.0
    PRESSURE_RANGE: float = 10.0
    COVERAGE_FRACTION: float = 0.9
    _VORONOI_GRID_STEP: float = 5.0  # pitch units between grid sample points

    # ── Public interface ──────────────────────────────────────────────────────

    def get_actions(self, state: GameState) -> TeamActions:
        gx, gy = goal_center()
        ball_loc = state["ball"]["location"]
        teammates = state["team"]
        opponents = state["opposition"]

        ball_carrier = next((p for p in teammates if p["has_ball"]), None)
        team_has_ball = ball_carrier is not None

        directions: dict[int, tuple[float, float]] = {}
        strike = False

        if team_has_ball:
            assert ball_carrier is not None
            directions, strike = self._in_possession_actions(
                ball_carrier, teammates, opponents, gx, gy
            )
        else:
            directions = self._out_of_possession_actions(teammates, opponents, ball_loc)

        carrier_num = ball_carrier["number"] if ball_carrier is not None else None
        return {
            "actions": {
                pid: {
                    "direction": direction,
                    "strike": strike and pid == carrier_num,
                }
                for pid, direction in directions.items()
            }
        }

    # ── In-possession helpers ─────────────────────────────────────────────────

    def _in_possession_actions(
        self,
        ball_carrier: PlayerState,
        teammates: list[PlayerState],
        opponents: list[PlayerState],
        gx: float,
        gy: float,
    ) -> tuple[dict[int, tuple[float, float]], bool]:
        all_locs = [p["location"] for p in teammates + opponents]
        centroids = self._voronoi_centroids(all_locs)

        directions: dict[int, tuple[float, float]] = {}
        for i, p in enumerate(teammates):
            cx, cy = centroids[i]
            px, py = p["location"]
            directions[p["number"]] = norm(cx - px, cy - py)

        bx, by = ball_carrier["location"]
        strike = False

        if dist([bx, by], [gx, gy]) <= self.STRIKE_RANGE:
            directions[ball_carrier["number"]] = norm(gx - bx, gy - by)
            strike = True
        else:
            forward_target = self._delaunay_pass_target(
                ball_carrier, teammates, opponents, min_x=bx
            )
            under_pressure = any(
                dist(ball_carrier["location"], opp["location"]) <= self.PRESSURE_RANGE
                for opp in opponents
            )
            if forward_target is not None:
                tx, ty = forward_target["location"]
                directions[ball_carrier["number"]] = norm(tx - bx, ty - by)
                strike = True
            elif under_pressure:
                nearest_opp = min(
                    opponents,
                    key=lambda o: dist(ball_carrier["location"], o["location"]),
                )
                escape_dy = -10.0 if nearest_opp["location"][1] > by else 10.0
                directions[ball_carrier["number"]] = (10.0, escape_dy)
                strike = True

        return directions, strike

    # ── Out-of-possession helpers ─────────────────────────────────────────────

    def _out_of_possession_actions(
        self,
        teammates: list[PlayerState],
        opponents: list[PlayerState],
        ball_loc: list[float],
    ) -> dict[int, tuple[float, float]]:
        closest = min(teammates, key=lambda p: dist(p["location"], ball_loc))

        directions: dict[int, tuple[float, float]] = {}
        for p in teammates:
            px, py = p["location"]
            if p["number"] == closest["number"]:
                directions[p["number"]] = norm(ball_loc[0] - px, ball_loc[1] - py)
            else:
                owned = self._zonal_opponents(p["number"], opponents)
                if owned and p["number"] != 1:  # Prevent GK from marking anyone
                    target = min(owned, key=lambda o: o["location"][0])
                    tx = ball_loc[0] + self.COVERAGE_FRACTION * (
                        target["location"][0] - ball_loc[0]
                    )
                    ty = ball_loc[1] + self.COVERAGE_FRACTION * (
                        target["location"][1] - ball_loc[1]
                    )
                    directions[p["number"]] = norm(tx - px, ty - py)
                else:
                    fp = self.formation.get(p["number"], [px, py])
                    directions[p["number"]] = norm(fp[0] - px, fp[1] - py)

        return directions

    # ── Geometry utilities ────────────────────────────────────────────────────

    def _voronoi_centroids(self, locations: list[list[float]]) -> list[list[float]]:
        """Approximate the centroid of each player's Voronoi cell via grid sampling."""
        W, H = STANDARD_PITCH_WIDTH, STANDARD_PITCH_HEIGHT
        step = self._VORONOI_GRID_STEP
        n = len(locations)
        sum_x = [0.0] * n
        sum_y = [0.0] * n
        count = [0] * n

        gx = 0.0
        while gx <= W:
            gy = 0.0
            while gy <= H:
                best, best_d2 = 0, float("inf")
                for i, loc in enumerate(locations):
                    d2 = (gx - loc[0]) ** 2 + (gy - loc[1]) ** 2
                    if d2 < best_d2:
                        best_d2, best = d2, i
                sum_x[best] += gx
                sum_y[best] += gy
                count[best] += 1
                gy += step
            gx += step

        return [
            [sum_x[i] / count[i], sum_y[i] / count[i]]
            if count[i] > 0
            else list(locations[i])
            for i in range(n)
        ]

    def _zonal_opponents(
        self, player_number: int, opponents: list[PlayerState]
    ) -> list[PlayerState]:
        """Return opponents whose location falls in ``player_number``'s formation zone."""
        if player_number not in self.formation:
            return []
        return [
            opp
            for opp in opponents
            if min(
                self.formation,
                key=lambda num: dist(opp["location"], self.formation[num]),
            )
            == player_number
        ]

    def _delaunay_pass_target(
        self,
        carrier: PlayerState,
        teammates: list[PlayerState],
        opponents: list[PlayerState],
        min_x: float | None = None,
    ) -> PlayerState | None:
        """Return the furthest-forward teammate connected to the carrier by a Delaunay edge.

        All ten players are triangulated together so the Delaunay graph reflects
        actual on-pitch proximity rather than a teammates-only subgraph.

        Parameters
        ----------
        min_x:
            When provided, only teammates with x > ``min_x`` are considered.
            Pass the carrier's x to restrict to players further forward.
        """
        all_players = teammates + opponents
        if len(all_players) < 3:
            return None

        points = np.array([p["location"] for p in all_players], dtype=float)
        try:
            tri = Delaunay(points)
        except Exception:
            return None

        carrier_idx = next(
            (
                i
                for i, p in enumerate(all_players)
                if p["is_teammate"] and p["number"] == carrier["number"]
            ),
            None,
        )
        if carrier_idx is None:
            return None

        neighbours: set[int] = set()
        for simplex in tri.simplices:
            if carrier_idx in simplex:
                neighbours.update(simplex)
        neighbours.discard(carrier_idx)

        connected_teammates = [
            all_players[i]
            for i in neighbours
            if all_players[i]["is_teammate"]
            and all_players[i]["number"] != carrier["number"]
            and (min_x is None or all_players[i]["location"][0] > min_x)
        ]
        if not connected_teammates:
            return None

        return max(connected_teammates, key=lambda p: p["location"][0])
