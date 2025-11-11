import os
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from P02_MSIE.T04_class_env.RL_envV2 import AIMH_ENV, CustomCallback
from P02_MSIE.T04_class_env.problem_sets import load_vrp


LEARN_TIMESTEPS = 5000
PROBLEM_SET = "LARGE"

vrptw = load_vrp(problem_set=PROBLEM_SET, verbose=0)


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
