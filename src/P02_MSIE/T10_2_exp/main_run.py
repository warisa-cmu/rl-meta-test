import os
import pathlib
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from P02_MSIE.T10_2_exp.env_v6 import AIMH_ENV, CustomCallback
from P02_MSIE.T10_2_exp.problem_sets import load_vrp

# RUN_TYPE = "NEW"
# LEARN_TIMESTEPS = 10000
# PROBLEM_SET = "LARGE"
# LOAD_FOLDER = ""
# LOAD_BEST_TYPE = ""
# LOAD_IT = 0


RUN_TYPE = "LOAD"
LEARN_TIMESTEPS = 20000
PROBLEM_SET = "LARGE"
LOAD_FOLDER = "R_20251115_161054"
LOAD_BEST_TYPE = "val"
LOAD_IT = 28010

SOLUTION_INJECT = [
    0.4850219,
    0.46107857,
    0.35311099,
    0.4403681,
    0.33899793,
    0.40565165,
    0.36144476,
    0.36183772,
    0.3759202,
    0.68005783,
    0.36384064,
    0.21889435,
    0.0,
    0.1806433,
    0.13151103,
    0.17445184,
    0.64024643,
    0.0,
    0.08947138,
    0.48511265,
    0.63355345,
    0.62620256,
    0.57660718,
    0.56454745,
    0.56594241,
    0.65900464,
    0.52450481,
    0.40635442,
    0.63920492,
    0.64083804,
    0.50566045,
    0.9878937,
    0.81658553,
    0.76490789,
    0.89412785,
    0.34060784,
    0.56820716,
    0.47235062,
    0.44935418,
    0.53568672,
    0.20753963,
    0.35353884,
    0.62293417,
    0.39096919,
    0.40659696,
    0.5631624,
    0.54257329,
    0.32328642,
    0.60455188,
    0.54784437,
]


vrptw = load_vrp(problem_set=PROBLEM_SET, verbose=0, solution_inject=SOLUTION_INJECT)
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
    model = SAC.load(
        f"{CURRENT_DIR}/save_models/{folder}/{best_type}_{it:05d}", env=env
    )

logger_custom = configure(log_dir, format_strings=["csv"])
model.set_logger(logger_custom)
custom_callback = CustomCallback(
    check_freq=1, save_dir=f"{CURRENT_DIR}/save_models", date_prefix=date_prefix
)
model.learn(total_timesteps=LEARN_TIMESTEPS, callback=custom_callback)
