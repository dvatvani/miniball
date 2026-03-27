from miniball.ai.interface import BaseAI, GameState, TeamActions
from miniball.ai.utils import dist, goal_center, relative_position


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
        ball_location = state["ball"]["location"]

        teammate_has_ball = any(
            p["is_teammate"] and p["has_ball"] for p in state["team"]
        )

        directions: dict[int, tuple[float, float]] = {}
        ball_carrier_number: int | None = None
        strike = False

        for p in state["team"]:
            player_number = p["number"]
            player_location = p["location"]

            if p["has_ball"]:
                # ── Dribble toward goal; strike when close enough ───────────
                directions[player_number] = relative_position(
                    player_location, goal_center()
                )
                ball_carrier_number = player_number
                strike = dist(player_location, goal_center()) < self.STRIKE_RANGE

            elif teammate_has_ball:
                # ── Drift back to formation position to open up space ────────
                formation_location = self.formation.get(player_number, player_location)
                directions[player_number] = relative_position(
                    player_location, formation_location
                )

            else:
                # ── Press toward the ball ──────────────────────────────────
                directions[player_number] = relative_position(
                    player_location, ball_location
                )

        return {
            "actions": {
                player_number: {
                    "direction": direction,
                    "strike": strike and player_number == ball_carrier_number,
                }
                for player_number, direction in directions.items()
            }
        }
