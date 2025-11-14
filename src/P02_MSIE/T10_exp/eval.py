import os
import pathlib
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC

from P02_MSIE.T09_exp.env_v6 import AIMH_ENV
from P02_MSIE.T09_exp.problem_sets import load_vrp
from P02_MSIE.T10_exp.conventional_agent import ConventionalAgent

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
PROBLEM_SET = "LARGE"
# RANDOM_SEED = 42
folder_sets = [
    dict(
        run_name="R1",
        folder="R_20251113_185747",
        file_type="end",
        it=200000,
    ),
]

vrptw = load_vrp(problem_set=PROBLEM_SET, verbose=0)

datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
#
fs = folder_sets[0]
#
folder = fs["folder"]
file_type = fs["file_type"]
it = fs["it"]
run_name = fs["run_name"]
print(f"Evaluating folder: {folder}, file_type: {file_type}, it: {it}")


# Load the model
# MODEL_NAME = "RL_RANDOM_TRAIN"
# model = SAC.load(f"{CURRENT_DIR}/save_models/{folder}/{file_type}_{it:05d}")

MODEL_NAME = "CA6"
model = ConventionalAgent(type_agent=MODEL_NAME)

# Create the environment
env = AIMH_ENV(vrp=vrptw)

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
