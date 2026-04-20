import math
from collections.abc import Sequence

import numpy as np

from miniball.ai.interface import BaseAI, GameState, PlayerState, TeamActions
from miniball.ai.utils.geometry import players_bounded_voronoi, players_in_polygon
from miniball.config import (
    BALL_RADIUS,
    PLAYER_SPEED,
    STANDARD_GOAL_HEIGHT,
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    MAX_STRIKE_SPEED,
)
from miniball.geometry import OPPOSITION_GOAL_CENTER, dist


class BaselineAI(BaseAI):
    """AI that uses Voronoi space creation and passing-lane checks.

    In possession
    ─────────────
    All players move toward the centroid of their Voronoi cell (computed from
    all ten players' current positions) to attempt to create space.

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

    STRIKE_RANGE: float = 20.0
    PRESSURE_RANGE: float = 10.0
    COVERAGE_FRACTION: float = 0.9
    PASSING_LANE_ANGLE: float = abs(
        math.atan2(PLAYER_SPEED * 1.2, MAX_STRIKE_SPEED * 0.5)
    )  # angle for pass lane check
    _VORONOI_GRID_STEP: float = 5.0  # pitch units between grid sample points

    # ── Public interface ──────────────────────────────────────────────────────

    def get_actions(self, state: GameState) -> TeamActions:
        directions: dict[int, tuple[float, float]] = {}
        strike = False

        if state.team_has_ball:
            directions, strike = self._in_possession_actions(state)
        else:
            directions = self._out_of_possession_actions(state)
            strike = False

        return {
            pid: {
                "direction": direction,
                "strike": strike
                and state.ball_carrier is not None
                and pid == state.ball_carrier.number,
            }
            for pid, direction in directions.items()
        }

    # ── In-possession helpers ─────────────────────────────────────────────────

    def _in_possession_actions(
        self,
        state: GameState,
    ) -> tuple[dict[int, tuple[float, float]], bool]:
        assert state.ball_carrier is not None
        directions: dict[int, tuple[float, float]] = {}
        strike = False

        # Move players to the centroid of player's current Voronoi region to find space
        vor = players_bounded_voronoi(state.team)
        for p, centroid in zip(state.team, vor.region_centroids):
            directions[p.number] = p.direction_to(centroid)

        # Move the GK towards the starting GK position
        gk = state.players(teammates=True, opposition=False, include_outfield=False)[0]
        directions[gk.number] = gk.direction_to(self.formation[gk.number])

        # Override the actions above for the ball carrier...

        # Dribble toward the opposition goal
        directions[state.ball_carrier.number] = state.ball_carrier.direction_to(
            OPPOSITION_GOAL_CENTER
        )
        # If there is an open forward teammate, pass to them
        forward_target = self._find_pass_target(state, forward_only=True)
        if forward_target is not None:
            directions[state.ball_carrier.number] = state.ball_carrier.direction_to(
                forward_target.location
            )
            strike = True

        # When under pressure, try to pass to pass the ball to an open teammate.
        # At this point, forward passes are preferred, but backwards passes are also allowed.
        # If no forward passes are available, the ball is cleared.
        nearest_opp = state.ball_carrier.closest_player(state.opposition)
        under_pressure = state.ball_carrier.dist_to(nearest_opp) <= self.PRESSURE_RANGE
        if under_pressure:
            forward_target = self._find_pass_target(state)
            if forward_target is not None:
                directions[state.ball_carrier.number] = state.ball_carrier.direction_to(
                    forward_target.location
                )
                strike = True
            else:
                # If there is no clear passing lane, clear the ball.
                nearest_opp_dy = (
                    nearest_opp.location[1] - state.ball_carrier.location[1]
                )
                directions[state.ball_carrier.number] = (
                    10,
                    -10 if nearest_opp_dy > 0 else 10,
                )
                strike = True

        # If close enough to the opposition goal, try to shoot
        if state.ball_carrier.dist_to(OPPOSITION_GOAL_CENTER) <= self.STRIKE_RANGE:
            shot_placement_location = self._determine_shot_placement(state)

            directions[state.ball_carrier.number] = state.ball_carrier.direction_to(
                shot_placement_location
            )

            strike = True

        return directions, strike

    # ── Out-of-possession helpers ─────────────────────────────────────────────

    def _out_of_possession_actions(
        self,
        state: GameState,
    ) -> dict[int, tuple[float, float]]:
        directions: dict[int, tuple[float, float]] = {}

        closest = state.ball.fastest_interceptor(state.team)
        for p in state.team:
            # Move the closest player toward the ball
            if p == closest:
                directions[p.number] = p.direction_to(p.intercept_point(state.ball))
                # If the closest player to the ball is the GK, but the GK isn't
                # going to get to the ball first, then move the GK back to the line.
                if (
                    closest.number == 1
                    and state.ball.fastest_interceptor(state.all_players) != closest
                ):
                    directions[p.number] = p.direction_to(self.formation[p.number])
                # If the projected interception point is past the goal line, then move
                # the player towards the point at which the ball would cross the goal line.
                if p.intercept_point(state.ball)[0] < 0:
                    cross = state.ball.position_when_crossing_x(0)
                    if cross is not None:
                        directions[p.number] = p.direction_to((0, cross[1]))

            # move other players toward their zonal marking target
            else:
                owned = self._zonal_opponents(p.number, state.opposition)
                if owned and p.number != 1:  # Prevent GK from marking anyone
                    target = min(owned, key=lambda o: o.location[0])
                    cover_point = (
                        state.ball.location[0]
                        + self.COVERAGE_FRACTION
                        * (target.location[0] - state.ball.location[0]),
                        state.ball.location[1]
                        + self.COVERAGE_FRACTION
                        * (target.location[1] - state.ball.location[1]),
                    )
                    directions[p.number] = p.direction_to(cover_point)
                else:
                    fp = self.formation.get(p.number, p.location)
                    directions[p.number] = p.direction_to(fp)

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
                key=lambda num: dist(opp.location, self.formation[num]),
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

    def _find_pass_target(
        self,
        state: GameState,
        forward_only: bool = False,
    ) -> PlayerState | None:
        """Return the furthest-forward teammate reachable via a clear passing lane.

        For each teammate a triangular corridor is built with
        ``_build_passing_lane_polygon``.  A lane is *clear* when no opponent
        falls inside it.  The most forward teammate with a clear lane is
        returned, or ``None`` if no clear lane exists.
        """
        assert state.ball_carrier is not None
        candidates = state.players(
            teammates=True,
            opposition=False,
            player_on_ball=False,
            players_on_cooldown=False,
        )
        if forward_only:
            candidates = [
                p for p in candidates if p.location[0] > state.ball_carrier.location[0]
            ]
        if not candidates:
            return None

        # Evaluate most-forward candidates first for an early exit
        candidates.sort(key=lambda p: p.location[0], reverse=True)

        for target in candidates:
            polygon = self._build_passing_lane_polygon(
                state.ball_carrier.location, target.location, self.PASSING_LANE_ANGLE
            )
            if not players_in_polygon(state.opposition, polygon):
                return target

        return None

    def _determine_shot_placement(self, state: GameState) -> tuple[float, float]:
        """Return the location to shoot the ball towards.

        For each candidate (top and bottom corners of the goal mouth), compute
        the smallest angle ``candidate – ball – opponent`` across all opponents.
        The candidate whose minimum clearance angle is largest is chosen — i.e.
        the shot with the widest angular gap past all defenders.
        """
        assert state.ball_carrier is not None
        shot_placement_candidates: list[tuple[float, float]] = [
            (
                STANDARD_PITCH_WIDTH,
                STANDARD_PITCH_HEIGHT / 2
                + STANDARD_GOAL_HEIGHT / 2
                - BALL_RADIUS * 1.5,
            ),
            (
                STANDARD_PITCH_WIDTH,
                STANDARD_PITCH_HEIGHT / 2
                - STANDARD_GOAL_HEIGHT / 2
                + BALL_RADIUS * 1.5,
            ),
        ]

        bx, by = state.ball_carrier.location

        def min_clearance_angle(candidate: tuple[float, float]) -> float:
            """Smallest angle between the ball→candidate ray and any ball→opponent ray."""
            cx, cy = candidate
            dcx, dcy = cx - bx, cy - by
            len_c = math.hypot(dcx, dcy)
            if len_c < 1e-9:
                return 0.0
            min_angle = math.pi
            for opp in state.opposition:
                ox, oy = opp.location
                dox, doy = ox - bx, oy - by
                len_o = math.hypot(dox, doy)
                if len_o < 1e-9:
                    return 0.0  # opponent at ball position: no clearance
                cos_a = (dcx * dox + dcy * doy) / (len_c * len_o)
                min_angle = min(min_angle, math.acos(max(-1.0, min(1.0, cos_a))))
            return min_angle

        return max(shot_placement_candidates, key=min_clearance_angle)
