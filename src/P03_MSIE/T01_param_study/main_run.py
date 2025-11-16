import json
import os
import pathlib
from dataclasses import asdict
from itertools import product

import pandas as pd
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

RUN_NAME = "RL vs Fixed Params Evaluation"
RUN_DESCRIPTION = "Experiment to evaluate RL model against fixed parameter settings on large VRPTW instances."
PROBLEM_SET = "LARGE"
LOAD_FOLDER = "R_20251116_091553"
FILE_PREFIX = "int"
LOAD_IT = 63924

F_range = [-10, 5, 2.5, 0, 2.5, 5, 10]
CR_range = [0.2, 0.5, 0.8]
MG_range = [0.2, 0.5, 0.8]

# VRPTW parameters
POPULATION_SIZE = 20

# RL parameters
PATIENCE = 800
VERBOSE = 0

# Not used in experiment, kept for compatibility
RUN_TYPE = ""
LEARN_TIMESTEPS = 0
SAVE_INTERVAL_SECONDS = 0


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

# Initialize the model
vrptw = load_vrptw(vpr_input_params, rl_input_params)

# Load existing model if RUN_TYPE is LOAD
env = RLMH_ENV(vrp=vrptw)
env = Monitor(
    env,
    filename=sim_input_params.monitor_filepath,
)

# Load trained RL model
folder = sim_input_params.load_folder
file_prefix = sim_input_params.file_prefix
it = sim_input_params.load_it
modelRL = SAC.load(
    f"{sim_input_params.current_dir}/saved_models/{folder}/{file_prefix}_{it:05d}_model",
    env=env,
)
# Load fixed param model


# Save all pamaters to a file
all_params = dict(
    **asdict(vpr_input_params), **asdict(rl_input_params), **asdict(sim_input_params)
)
all_params["current_dir"] = str(all_params["current_dir"])
output_params_dir = f"{sim_input_params.current_dir}/params"
os.makedirs(output_params_dir, exist_ok=True)
with open(f"{output_params_dir}/{sim_input_params.date_prefix}_params.json", "w") as f:
    json.dump(all_params, f, indent=4)


# Run the evaluation

data_stored = []
for seed in range(100):
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    data_array = []
    reward_total = 0
    idx = -1
    while not (terminated or truncated):
        idx += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        data_added = {**info, "reward": reward, "action": action}
        data_array.append(data_added)
    df = pd.DataFrame.from_dict(data_array)
    df["seed"] = seed
    df["model_name"] = MODEL_NAME
    best_solution = df["best_solution"].min() * vrptw.solution_scale_factor
    data_stored.append(
        {
            "seed": seed,
            "best_solution": best_solution,
            "df": df,
            "global_solution_history": env.vrp.global_solution_history,
            "fitness_trial_history": env.vrp.fitness_trial_history,
        }
    )
    print(
        f"Model Name {MODEL_NAME}, Evaluating seed: {seed}, best solution: {best_solution}"
    )

# Save the evaluation results
os.makedirs(f"{CURRENT_DIR}/eval_results", exist_ok=True)
with open(
    f"{CURRENT_DIR}/eval_results/{MODEL_NAME}_{datetime_now}.pkl",
    "wb",
) as f:
    pickle.dump(data_stored, f)
