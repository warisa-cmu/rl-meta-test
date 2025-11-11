import os
from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from P02_MSIE.T04_class_env.DE_IM_VRPTW_classV7 import VRPTW
from P02_MSIE.T04_class_env.RL_envV2 import AIMH_ENV, CustomCallback

LEARN_TIMESTEPS = 10

distance = (
    pd.read_excel(
        r"./src/Source/rl_meta_test_data_25_customer.xlsx",
        sheet_name="distance",
    )
    .fillna(9999999)
    .to_numpy()
)

df_vehicle = (
    pd.read_excel(
        r"./src/Source/rl_meta_test_data_25_customer.xlsx",
        sheet_name="vehicle",
    )
    .iloc[:, :2]
    .to_numpy(dtype=int)
)
vehicle = df_vehicle[0]

df_101 = pd.read_excel(
    r"./src/Source/rl_meta_test_data_25_customer.xlsx",
    sheet_name="customer",
).iloc[:, 3:]
demand = df_101.iloc[:, 0].to_numpy()
readyTime = df_101.iloc[:, 1].to_numpy()
dueDate = df_101.iloc[:, 2].to_numpy()
serviceTime = df_101.iloc[:, 3].to_numpy()

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
patience = 200
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

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filepath = f"{log_dir}/{date_prefix}_training.log"
csv_filepath_default = f"{log_dir}/progress.csv"
monitor_filepath = f"{log_dir}/{date_prefix}_monitor.csv"

env = AIMH_ENV(vrp=vrptw)
env = Monitor(
    env,
    filename=monitor_filepath,
)
model = SAC("MlpPolicy", env, verbose=1)
logger_custom = configure(log_dir, format_strings=["csv"])
model.set_logger(logger_custom)
custom_callback = CustomCallback(
    check_freq=1, save_dir="./save_models", date_prefix=date_prefix
)
model.learn(total_timesteps=LEARN_TIMESTEPS, callback=custom_callback)
