from miniball.ai.interface import BaseAI, GameState, TeamActions
from miniball.ai.utils import opposition_goal_center


class BallChasersAI(BaseAI):
    """Simple rule-based AI.

    Decision hierarchy (evaluated per player, per frame):

    1. **Has the ball** → dribble straight toward the attacking goal;
       strike when within ``STRIKE_RANGE`` normalised units of the goal centre.
    2. **No ball / opposition has it** → press toward the ball at full speed.
    3. **Teammate has the ball** → drift back toward formation position to avoid
       crowding the ball carrier and leave space open.

    Because the state is always normalised to attack right and expressed in
    standard pitch coordinates, this class contains no team-side, pixel, or
    coordinate-direction logic.
    """

    STRIKE_RANGE: float = (
        28.0  # normalised units to goal centre at which the AI strikes
    )

    def get_actions(self, state: GameState) -> TeamActions:
        directions: dict[int, tuple[float, float]] = {}
        ball_carrier_number: int | None = None
        strike = False

        for p in state.team:
            if p.has_ball:
                # ── Dribble toward goal; strike when close enough ───────────
                directions[p.number] = p.direction_to(opposition_goal_center())
                ball_carrier_number = p.number
                strike = p.dist_to(opposition_goal_center()) < self.STRIKE_RANGE

            elif state.team_has_ball:
                # ── Drift back to formation position to open up space ────────
                formation_location = self.formation.get(p.number, p.location)
                directions[p.number] = p.direction_to(formation_location)

            else:
                # ── Press toward the ball ──────────────────────────────────
                directions[p.number] = p.direction_to(state.ball.location)

        return {
            player_number: {
                "direction": direction,
                "strike": strike and player_number == ball_carrier_number,
            }
            for player_number, direction in directions.items()
        }
