from miniball.simulation import GameSimulation
from miniball.teams import teams

sim = GameSimulation(teams["Baseline (1-2-2)"], teams["Baseline (1-3-1)"])
df = sim.simulate_match()
sim.export_history()
