from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC

from P02_MSIE.T03_class_env.DE_IM_VRPTW_classV6 import (
    VRPTW,
)
from P02_MSIE.T03_class_env.RL_envV1 import AIMH_ENV

distance = (
    pd.read_excel(r"./src/Source/rl_meta_test_data.xlsx", sheet_name="distance")
    .fillna(9999999)
    .to_numpy()
)

df_vehicle = (
    pd.read_excel(r"./src/Source/rl_meta_test_data.xlsx", sheet_name="vehicle")
    .iloc[:, :2]
    .to_numpy(dtype=int)
)
vehicle = df_vehicle[0]

df_101 = pd.read_excel(
    r"./src/Source/rl_meta_test_data.xlsx", sheet_name="customer"
).iloc[:, 3:]
demand = df_101.iloc[:, 0].to_numpy()
readyTime = df_101.iloc[:, 1].to_numpy()
dueDate = df_101.iloc[:, 2].to_numpy()
serviceTime = df_101.iloc[:, -1].to_numpy()

kwargs = {
    "distance": distance,
    "demand": demand,
    "readyTime": readyTime,
    "dueDate": dueDate,
    "serviceTime": serviceTime,
    "vehicle": vehicle,
}
dimensions = len(distance) - 1 + vehicle[0]
interval_it = 20
patience = 4000
population_size = 4
bounds = np.array([[0, 1]] * dimensions)

vrptw = VRPTW(
    population_size=population_size,
    dimensions=dimensions,
    bounds=bounds,
    distance=distance,
    demand=demand,
    readyTime=readyTime,
    dueDate=dueDate,
    serviceTime=serviceTime,
    vehicle=vehicle,
    interval_it=interval_it,
    patience=patience,
    target_solution=48,
)


# folder = "R_20251111_120311"
folder = "R_20251111_122008"
best_type = "val"
it = 68


model = SAC.load(f"save_models/{folder}/{best_type}_{it:05d}")
env = AIMH_ENV(vrp=vrptw)
obs, info = env.reset()
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
