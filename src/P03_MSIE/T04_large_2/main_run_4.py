import json
import pathlib
from dataclasses import asdict
import os
from stable_baselines3 import SAC
from stable_baselines3.common.logger import CSVOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor

from rlmh_v1.env import RLMH_ENV, SIM_INPUT_PARAMS, CustomCallback
from rlmh_v1.utils import LinearScaler, RewardParams
from rlmh_v1.vrptw import (
    RL_INPUT_PARAMS,
    VRPTW_INPUT_PARAMS,
    load_vrptw,
)

RUN_NAME = "T04_Large_2"
RUN_DESCRIPTION = "New large problem run"
RUN_TYPE = "NEW"
LEARN_TIMESTEPS = 400000
PROBLEM_SET = "LARGE2"
SAVE_INTERVAL_SECONDS = 8 * 60
LOAD_FOLDER = ""
FILE_PREFIX = ""
LOAD_IT = 0


# RUN_NAME = ""
# RUN_DESCRIPTION = ""
# RUN_TYPE = "LOAD"
# LEARN_TIMESTEPS = 2000
# PROBLEM_SET = "LARGE"
# LOAD_FOLDER = "R_20251116_085826"
# FILE_PREFIX = "int"
# LOAD_IT = 834
# SAVE_INTERVAL_SECONDS = 8 * 60  # 8 minutes

# VRPTW parameters
POPULATION_SIZE = 20

# RL parameters
PATIENCE = 800
VERBOSE = 0


vpr_input_params = VRPTW_INPUT_PARAMS(
    problem_set=PROBLEM_SET,  # Options: "SMALL", "LARGE"
    population_size=POPULATION_SIZE,
)

sim_input_params = SIM_INPUT_PARAMS(
    run_type=RUN_TYPE,
    current_dir=pathlib.Path(__file__).parent.resolve(),
    save_interval_seconds=SAVE_INTERVAL_SECONDS,
    learn_timesteps=LEARN_TIMESTEPS,
    load_folder=LOAD_FOLDER,
    file_prefix=FILE_PREFIX,
    load_it=LOAD_IT,
)


if vpr_input_params.problem_set == "SMALL":
    rl_input_params = RL_INPUT_PARAMS(
        sc_F=LinearScaler(
            bounds=(-10, 10), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_CR=LinearScaler(
            bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_MG=LinearScaler(
            bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_solution=LinearScaler(bounds=(0, 100), bounds_scaled=(0, 1)),
        sc_iteration=LinearScaler(bounds=(0, 1e5), bounds_scaled=(0, 10)),
        interval_it=10,
        target_solution=40,
        reward_params=RewardParams(
            reward_mode="TARGET_ENHANCED_3",
            alpha_target=1,
            alpha_patience=10,
            s=2.0,
            c=0.1,
        ),
        patience=PATIENCE,
        verbose=VERBOSE,
        convert_none_seed_to_number=False,
    )

elif vpr_input_params.problem_set == "LARGE":
    rl_input_params = RL_INPUT_PARAMS(
        sc_F=LinearScaler(
            bounds=(-10, 10), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_CR=LinearScaler(
            bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_MG=LinearScaler(
            bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_solution=LinearScaler(bounds=(0, 2000), bounds_scaled=(0, 2)),
        sc_iteration=LinearScaler(bounds=(0, 1e5), bounds_scaled=(0, 10)),
        interval_it=10,
        target_solution=190,
        reward_params=RewardParams(
            reward_mode="TARGET_ENHANCED_3",
            alpha_target=1,
            alpha_patience=10,
            s=2.0,
            c=0.1,
        ),
        patience=PATIENCE,
        verbose=VERBOSE,
        convert_none_seed_to_number=False,
    )
elif vpr_input_params.problem_set == "LARGE2":
    rl_input_params = RL_INPUT_PARAMS(
        sc_F=LinearScaler(
            bounds=(-10, 10), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_CR=LinearScaler(
            bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_MG=LinearScaler(
            bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5
        ),
        sc_solution=LinearScaler(bounds=(0, 2000), bounds_scaled=(0, 2)),
        sc_iteration=LinearScaler(bounds=(0, 1e5), bounds_scaled=(0, 10)),
        interval_it=10,
        target_solution=650,
        reward_params=RewardParams(
            reward_mode="TARGET_ENHANCED_3",
            alpha_target=1,
            alpha_patience=10,
            s=2.0,
            c=0.1,
        ),
        patience=PATIENCE,
        verbose=VERBOSE,
        convert_none_seed_to_number=False,
    )

# Initialize the model
vrptw = load_vrptw(vpr_input_params, rl_input_params)

# Load existing model if RUN_TYPE is LOAD
env = RLMH_ENV(vrp=vrptw)
env = Monitor(
    env,
    filename=sim_input_params.monitor_filepath,
)

# Initialize or load the model based on run_type
if sim_input_params.run_type == "NEW":
    model = SAC("MlpPolicy", env, verbose=1)
elif sim_input_params.run_type == "LOAD":
    folder = sim_input_params.load_folder
    file_prefix = sim_input_params.file_prefix
    it = sim_input_params.load_it
    model = SAC.load(
        f"{sim_input_params.current_dir}/saved_models/{folder}/{file_prefix}_{it:05d}_model",
        env=env,
    )

# Set up custom logger
logger = Logger(
    folder=sim_input_params.log_dir,
    output_formats=[CSVOutputFormat(sim_input_params.logger_filepath)],
)
model.set_logger(logger)

# Define and add the custom callback
custom_callback = CustomCallback(
    check_freq=1,
    save_dir=f"{sim_input_params.current_dir}/saved_models",
    date_prefix=sim_input_params.date_prefix,
    save_interval_seconds=sim_input_params.save_interval_seconds,
)

# Save all pamaters to a file
all_params = dict(
    **asdict(vpr_input_params), **asdict(rl_input_params), **asdict(sim_input_params)
)
all_params["current_dir"] = str(all_params["current_dir"])
output_params_dir = f"{sim_input_params.current_dir}/params"
os.makedirs(output_params_dir, exist_ok=True)
with open(f"{output_params_dir}/{sim_input_params.date_prefix}_params.json", "w") as f:
    json.dump(all_params, f, indent=4)


# Start training
model.learn(total_timesteps=sim_input_params.learn_timesteps, callback=custom_callback)
