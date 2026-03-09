from miniball.ai.helpers import BaseAI, GameState, TeamActions


class StationaryAI(BaseAI):
    """Every player stands still and never shoots.

    Useful as a neutral placeholder while you develop the real AI.
    """

    def get_actions(self, state: GameState) -> TeamActions:
        return {
            "actions": {
                p["number"]: {"direction": [0.0, 0.0], "shoot": False}
                for p in state["team"]
            }
        }
