from miniball.simulation import GameSimulation
from miniball.team_config import teams

sim = GameSimulation(teams["Baseline (1-2-2)"], teams["Baseline (1-3-1)"])
while not sim.game_over:
    sim.step(1 / 60)
sim.export_history()
