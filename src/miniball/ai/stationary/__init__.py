from miniball.ai.interface import BaseAI, GameState, TeamActions


class StationaryAI(BaseAI):
    """Every player stands still and never strikes.

    Useful as a neutral placeholder while you develop the real AI.
    """

    def get_actions(self, state: GameState) -> TeamActions:
        return {
            "actions": {
                p["number"]: {"direction": (0.0, 0.0), "strike": False}
                for p in state["team"]
            }
        }
