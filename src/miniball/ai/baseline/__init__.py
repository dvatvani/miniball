from collections.abc import Sequence

import numpy as np

from miniball.ai.interface import BaseAI, GameState, PlayerState, TeamActions
from miniball.ai.utils import dist, goal_center, norm
from miniball.ai.utils.geometry import players_bounded_voronoi, players_in_polygon


class BaselineAI(BaseAI):
    """AI that uses Voronoi space creation and passing-lane checks.

    In possession
    ─────────────
    All players move toward the centroid of their Voronoi cell (computed from
    all ten players' current positions) to spread out and attempt to create space.

    The ball carrier's decision overrides the space-creation movement:

    1. Within ``STRIKE_RANGE`` of the attacking goal → face goal and strike.
    2. A forward teammate exists with a clear passing lane (no opponent inside
       the triangular corridor defined by ``PASSING_LANE_ANGLE``) → pass to
       the most forward such teammate.
    3. An opponent within ``PRESSURE_RANGE`` and no clear forward pass →
       diagonal escape strike (clearance).
    4. Otherwise → continue moving toward own Voronoi centroid to create space.

    Out of possession
    ─────────────────
    1. The teammate nearest to the ball presses immediately.
    2. Every other player is assigned a Zonal marking zone based on the Voronoi
    regions from the team formation positions.
    3. If an opponent currently occupies a player's zone, that player moves to
       the point 90 % of the way along the line from the ball to the most
       forward (lowest x) opponent in the zone to cut off the pass lane.
    4. If the zone is empty, the player returns to their formation position.
    """

    STRIKE_RANGE: float = 40.0
    PRESSURE_RANGE: float = 10.0
    COVERAGE_FRACTION: float = 0.9
    PASSING_LANE_ANGLE: float = np.radians(20)  # half-cone angle for pass lane check
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
        players = teammates + opponents
        vor = players_bounded_voronoi(players)
        centroids = vor.region_centroids

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
            forward_target = self._passing_lane_pass_target(
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

    def _build_passing_lane_polygon(
        self,
        start: Sequence[float],
        end: Sequence[float],
        angle: float,
    ) -> np.ndarray:
        """Build a triangular passing-lane polygon.

        Returns a ``(3, 2)`` array representing triangle **ABC** where:

        * **A** is ``start``
        * the midpoint of **BC** is ``end``
        * the angle ∠BAC equals ``angle`` (radians)

        The triangle opens from the passer toward the target, so any player
        inside it lies within the corridor between the two points.
        """
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        dx, dy = ex - sx, ey - sy
        length = np.hypot(dx, dy)
        if length == 0.0:
            return np.array([[sx, sy], [ex, ey], [ex, ey]])

        # Unit vector perpendicular to the pass direction
        px, py = -dy / length, dx / length
        half_width = length * np.tan(angle / 2.0)

        A = np.array([sx, sy])
        B = np.array([ex + px * half_width, ey + py * half_width])
        C = np.array([ex - px * half_width, ey - py * half_width])
        return np.array([A, B, C])

    def _passing_lane_pass_target(
        self,
        carrier: PlayerState,
        teammates: list[PlayerState],
        opponents: list[PlayerState],
        min_x: float | None = None,
    ) -> PlayerState | None:
        """Return the furthest-forward teammate reachable via a clear passing lane.

        For each forward teammate a triangular corridor is built with
        ``_build_passing_lane_polygon``.  A lane is *clear* when no opponent
        falls inside it.  The most forward teammate with a clear lane is
        returned, or ``None`` if no clear lane exists.

        Parameters
        ----------
        min_x:
            When provided, only teammates with ``x > min_x`` are considered.
            Pass the carrier's x to restrict to players further forward.
        """
        candidates = [
            t
            for t in teammates
            if t["number"] != carrier["number"]
            and (min_x is None or t["location"][0] > min_x)
        ]
        if not candidates:
            return None

        # Evaluate most-forward candidates first for an early exit
        candidates.sort(key=lambda p: p["location"][0], reverse=True)

        for target in candidates:
            polygon = self._build_passing_lane_polygon(
                carrier["location"], target["location"], self.PASSING_LANE_ANGLE
            )
            if not players_in_polygon(opponents, polygon):
                return target

        return None
