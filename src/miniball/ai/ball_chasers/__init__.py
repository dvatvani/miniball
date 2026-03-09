from miniball.ai.helpers import BaseAI, GameState, TeamActions


class BallChasersAI(BaseAI):
    """Simple rule-based AI.

    Decision hierarchy (evaluated per player, per frame):

    1. **Has the ball** → dribble straight toward the attacking goal;
       shoot when within ``SHOOT_RANGE`` normalised units of the goal centre.
    2. **No ball / opposition has it** → press toward the ball at full speed.
    3. **Teammate has the ball** → drift back toward home position to avoid
       crowding the ball carrier and leave space open.

    Home positions are cached from each player's location on the first frame
    so the AI naturally inherits whatever starting layout the game uses.

    Because the state is always normalised to attack right and expressed in
    standard pitch coordinates, this class contains no team-side, pixel, or
    coordinate-direction logic.
    """

    SHOOT_RANGE: float = 28.0  # normalised units to goal centre at which the AI shoots
    HOME_DEADBAND: float = 2.0  # normalised units – don't move if already close to home

    def get_actions(self, state: GameState) -> TeamActions:
        gx, gy = self._goal_center()
        ball_loc = state["ball"]["location"]

        teammate_has_ball = any(
            p["is_teammate"] and p["has_ball"] for p in state["team"]
        )

        shoot = False
        directions = {}

        for p in state["team"]:
            pid = p["number"]
            px, py = p["location"]

            if p["has_ball"]:
                # ── Dribble toward goal; shoot when close enough ───────────
                dx, dy = self._norm(gx - px, gy - py)
                dist_to_goal = self._dist([px, py], [gx, gy])
                directions[pid] = [dx, dy]
                shoot = dist_to_goal < self.SHOOT_RANGE

            elif teammate_has_ball:
                # ── Drift back to home position to open up space ───────────
                formation_location = self.formation.get(pid, [px, py])
                if self._dist([px, py], formation_location) > self.HOME_DEADBAND:
                    dx, dy = self._norm(
                        formation_location[0] - px, formation_location[1] - py
                    )
                    directions[pid] = [dx, dy]
                else:
                    directions[pid] = [0.0, 0.0]

            else:
                # ── Press toward the ball ──────────────────────────────────
                dx, dy = self._norm(ball_loc[0] - px, ball_loc[1] - py)
                directions[pid] = [dx, dy]

        return {
            "directions": directions,
            "shoot": shoot,
        }
