import json
import os
import pathlib
import pickle
from dataclasses import asdict
from itertools import product

import pandas as pd
from stable_baselines3 import SAC

from P03_MSIE.T01_param_study.custom_agent import FixedParamAgent
from rlmh_v1.env import RLMH_ENV, SIM_INPUT_PARAMS
from rlmh_v1.utils import LinearScaler, RewardParams
from rlmh_v1.vrptw import (
    RL_INPUT_PARAMS,
    VRPTW_INPUT_PARAMS,
    load_vrptw,
)

# ----------------------- Start of Input -----------------------
RUN_NAME = "RL vs Fixed Params Evaluation"
RUN_DESCRIPTION = "Experiment to evaluate RL model against fixed parameter settings on large VRPTW instances."
PROBLEM_SET = "LARGE"
LOAD_FOLDER = "R_20251116_091553"
FILE_PREFIX = "int"
LOAD_IT = 63924
WITH_RL = False

# Fixed parameter ranges
F_range = [2.5]
CR_range = [
    0,
    0.5,
    1,
]
MG_range = [
    0,
    0.5,
    1,
]
# Number of repeats for each parameter setting
NUM_REPEATS = 20
# Scalers for parameters
sc_F = LinearScaler(bounds=(-10, 10), bounds_scaled=(-0.5, 0.5), starting_value=0.5)
sc_CR = LinearScaler(bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5)
sc_MG = LinearScaler(bounds=(0, 1), bounds_scaled=(-0.5, 0.5), starting_value=0.5)

# VRPTW parameters
POPULATION_SIZE = 20

# RL parameters
PATIENCE = 800
VERBOSE = 0

# Not used in experiment, kept for compatibility
RUN_TYPE = ""
LEARN_TIMESTEPS = 0
SAVE_INTERVAL_SECONDS = 0
# ----------------------- End of Input -----------------------

vpr_input_params = VRPTW_INPUT_PARAMS(
    problem_set=PROBLEM_SET,  # Options: "SMALL", "LARGE"
    population_size=POPULATION_SIZE,
)

sim_input_params = SIM_INPUT_PARAMS(
    run_name=RUN_NAME,
    run_description=RUN_DESCRIPTION,
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
        sc_F=sc_F,
        sc_CR=sc_CR,
        sc_MG=sc_MG,
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

# Create the environment
env = RLMH_ENV(vrp=vrptw)

# Set up models
models = []
model_names = []

if WITH_RL:
    # Load trained RL model
    folder = sim_input_params.load_folder
    file_prefix = sim_input_params.file_prefix
    it = sim_input_params.load_it
    model_RL = SAC.load(
        f"{sim_input_params.current_dir}/saved_models/{folder}/{file_prefix}_{it:05d}_model",
        env=env,
    )
    models.append(model_RL)
    model_names.append("RL")
else:
    # Load fixed param model
    for F, CR, MG in product(F_range, CR_range, MG_range):
        F_sc = sc_F.transform(F)
        CR_sc = sc_CR.transform(CR)
        MG_sc = sc_MG.transform(MG)
        model_name = f"({F},{CR},{MG})"
        model_fixed = FixedParamAgent(model_name, F_sc, CR_sc, MG_sc)
        models.append(model_fixed)
        model_names.append(model_name)


# Save all parameters to a file
all_params = dict(
    **asdict(vpr_input_params),
    **asdict(rl_input_params),
    **asdict(sim_input_params),
    F_range=F_range,
    CR_range=CR_range,
    MG_range=MG_range,
    NUM_REPEATS=NUM_REPEATS,
    WITH_RL=WITH_RL,
)
all_params["current_dir"] = str(all_params["current_dir"])
output_params_dir = f"{sim_input_params.current_dir}/params"
os.makedirs(output_params_dir, exist_ok=True)
with open(f"{output_params_dir}/{sim_input_params.date_prefix}_params.json", "w") as f:
    json.dump(all_params, f, indent=4)


# Run the evaluation
data_stored = []
for model, model_name in zip(models, model_names):
    for seed in range(NUM_REPEATS):
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
            data_added = {**info, "obs": obs, "reward": reward, "action": action}
            data_array.append(data_added)
        df = pd.DataFrame.from_dict(data_array)
        df["seed"] = seed
        df["model_name"] = model_name
        best_solution = df["best_solution"].min()
        data_stored.append(
            {
                "model_name": model_name,
                "seed": seed,
                "best_solution": best_solution,
                "df": df,
                "global_solution_history": env.vrp.global_solution_history,
                "fitness_trial_history": env.vrp.fitness_trial_history,
            }
        )
        print(
            f"Model Name {model_name}, Evaluating seed: {seed}, best solution: {best_solution}"
        )

# Save the evaluation results
os.makedirs(f"{sim_input_params.current_dir}/eval_results", exist_ok=True)
with open(
    f"{sim_input_params.current_dir}/eval_results/{sim_input_params.date_prefix}.pkl",
    "wb",
) as f:
    pickle.dump(data_stored, f)
