from miniball.simulation import GameSimulation
from miniball.team_config import teams

sim = GameSimulation(teams["Baseline AI (1-2-2)"], teams["Baseline AI (1-3-1)"])
while not sim.game_over:
    sim.step(1 / 60)
sim.export_history()
