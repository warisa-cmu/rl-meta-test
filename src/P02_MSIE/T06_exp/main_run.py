import os
import pathlib
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from P02_MSIE.T06_exp.env_v4 import AIMH_ENV, CustomCallback
from P02_MSIE.T06_exp.problem_sets import load_vrp

RUN_TYPE = "NEW"
LEARN_TIMESTEPS = 20000
PROBLEM_SET = "LARGE"
LOAD_FOLDER = ""
LOAD_BEST_TYPE = ""
LOAD_IT = 0


# RUN_TYPE = "LOAD"
# LEARN_TIMESTEPS = 100000
# PROBLEM_SET = "LARGE"
# LOAD_FOLDER = "R_20251112_204554"
# LOAD_BEST_TYPE = "end"
# LOAD_IT = 100000

vrptw = load_vrp(problem_set=PROBLEM_SET, verbose=0)
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
log_dir = f"{CURRENT_DIR}/logs"
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

if RUN_TYPE == "NEW":
    model = SAC("MlpPolicy", env, verbose=1)
elif RUN_TYPE == "LOAD":
    folder = LOAD_FOLDER
    best_type = LOAD_BEST_TYPE
    it = LOAD_IT
    model = SAC.load(f"save_models/{folder}/{best_type}_{it:05d}", env=env)

logger_custom = configure(log_dir, format_strings=["csv"])
model.set_logger(logger_custom)
custom_callback = CustomCallback(
    check_freq=1, save_dir=f"{CURRENT_DIR}/save_models", date_prefix=date_prefix
)
model.learn(total_timesteps=LEARN_TIMESTEPS, callback=custom_callback)
