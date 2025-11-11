from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC

from P02_MSIE.T04_class_env.RL_envV2 import AIMH_ENV
from P02_MSIE.T04_class_env.problem_sets import load_vrp


PROBLEM_SET = "LARGE"

vrptw = load_vrp(
    problem_set=PROBLEM_SET,
)
vrptw.patience = 1000

folder = "R_20251111_210321"
best_type = "end"
it = 1000


model = SAC.load(f"save_models/{folder}/{best_type}_{it:05d}")
env = AIMH_ENV(vrp=vrptw)
obs, info = env.reset(seed=42)
terminated = False
truncated = False
data_array = []
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    data_added = {**info, "reward": reward, "action": action}
    data_array.append(data_added)
df = pd.DataFrame.from_dict(data_array)
datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("_tmp", exist_ok=True)
df.to_excel(f"_tmp/test_load_output_{datetime_now}.xlsx", index=False)
console = Console()
table = Table(title="Data")
for col in df.columns:
    table.add_column(col, style="cyan", no_wrap=True)
for index, row in df.iterrows():
    table.add_row(*[str(item) for item in row])
console.print(table)

fig, ax = plt.subplots(1, figsize=(10, 5))
x = np.arange(vrptw.idx_iteration + 1)
y1 = vrptw.global_solution_history
y2 = vrptw.fitness_trial_history
ax.plot(x, y1, marker=".", label="Best Solution")
ax.plot(x, y2, marker=".", label="Fitness Trial")
ax.set(
    xlabel="iteration",
    ylabel="Total Distance (Km.)",
    title="Differential Evoluation + Island Model Algorithm Replication",
)
ax.legend()
plt.show()
