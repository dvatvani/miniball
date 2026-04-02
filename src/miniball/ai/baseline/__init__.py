from collections.abc import Sequence

import numpy as np

from miniball.ai.interface import BaseAI, GameState, PlayerState, TeamActions
from miniball.ai.utils import dist, opposition_goal_center, player_closest_to_point
from miniball.ai.utils.geometry import players_bounded_voronoi, players_in_polygon


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

    STRIKE_RANGE: float = 40.0
    PRESSURE_RANGE: float = 10.0
    COVERAGE_FRACTION: float = 0.9
    PASSING_LANE_ANGLE: float = np.radians(20)  # half-cone angle for pass lane check
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
                "strike": strike and pid == state.ball_carrier.number,
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
        gk = state.players(teammates=True, include_goalkeepers=True)[0]
        directions[gk.number] = gk.direction_to(self.formation[gk.number])

        # Strike if close enough to the goal
        if state.ball_carrier.dist_to(opposition_goal_center()) <= self.STRIKE_RANGE:
            directions[state.ball_carrier.number] = state.ball_carrier.direction_to(
                opposition_goal_center()
            )
            strike = True

        # Otherwise, look for a clear passing lane, and clear the ball if under pressure
        else:
            forward_target = self._passing_lane_pass_target(
                state.ball_carrier,
                state.team,
                state.opposition,
                min_x=state.ball_carrier.location[0],
            )
            under_pressure = (
                state.ball_carrier.dist_to(
                    state.ball_carrier.closest_player(state.opposition).location
                )
                <= self.PRESSURE_RANGE
            )
            if forward_target is not None:
                directions[state.ball_carrier.number] = state.ball_carrier.direction_to(
                    forward_target
                )
                strike = True
            elif under_pressure:
                nearest_opp = player_closest_to_point(
                    state.opposition, state.ball_carrier.location
                )
                nearest_opp_dy = (
                    nearest_opp.location[1] - state.ball_carrier.location[1]
                )
                directions[state.ball_carrier.number] = (
                    10,
                    -10 if nearest_opp_dy > 0 else 10,
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

            # move other players toward their zonal marking target
            else:
                owned = self._zonal_opponents(p.number, state.opposition)
                if owned and p.number != 1:  # Prevent GK from marking anyone
                    target = min(owned, key=lambda o: o.location[0])
                    tx = state.ball.location[0] + self.COVERAGE_FRACTION * (
                        target.location[0] - state.ball.location[0]
                    )
                    ty = state.ball.location[1] + self.COVERAGE_FRACTION * (
                        target.location[1] - state.ball.location[1]
                    )
                    directions[p.number] = (tx - p.location[0], ty - p.location[1])
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
            if t.number != carrier.number and (min_x is None or t.location[0] > min_x)
        ]
        if not candidates:
            return None

        # Evaluate most-forward candidates first for an early exit
        candidates.sort(key=lambda p: p.location[0], reverse=True)

        for target in candidates:
            polygon = self._build_passing_lane_polygon(
                carrier.location, target.location, self.PASSING_LANE_ANGLE
            )
            if not players_in_polygon(opponents, polygon):
                return target

        return None
