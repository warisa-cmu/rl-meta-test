import os
import pathlib
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import SAC

from P02_MSIE.T09_exp.env_v6 import AIMH_ENV

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
PROBLEM_SET = "LARGE"
RANDOM_SEED = 42
folder_sets = [
    dict(
        run_name="R1",
        folder="R_20251113_213604",
        file_type="val",
        it=34586,
    ),
]

datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")

for fs in folder_sets:
    folder = fs["folder"]
    file_type = fs["file_type"]
    it = fs["it"]
    run_name = fs["run_name"]
    print(f"Evaluating folder: {folder}, file_type: {file_type}, it: {it}")

    # Load experiences
    with open(f"{CURRENT_DIR}/save_models/{folder}/exp_{it:05d}.pkl", "rb") as f:
        exp = pickle.load(f)

    with open(f"{CURRENT_DIR}/save_models/{folder}/vrp_{it:05d}.pkl", "rb") as f:
        info = pickle.load(f)

    vrptw = info["vrptw"]

    # Load the model
    model = SAC.load(f"{CURRENT_DIR}/save_models/{folder}/{file_type}_{it:05d}")

    # Create the environment
    env = AIMH_ENV(vrp=vrptw)

    # Run the experience
    obs, info = env.reset(seed=RANDOM_SEED)
    terminated = False
    truncated = False
    data_array = []
    reward_total = 0
    idx = -1
    for action in exp["action"]:
        idx += 1
        obs, reward, terminated, truncated, info = env.step(action)
        # if idx % 100 == 0:
        #     print(f"Step: {idx}, Reward: {reward:.2f}, Action: {action}, Info: {info}")
        reward_total += reward
        data_added = {**info, "reward": reward, "action": action}
        data_array.append(data_added)
    df = pd.DataFrame.from_dict(data_array)
    best_solution = df["best_solution"].min() * vrptw.solution_scale_factor

    # Save results
    os.makedirs(f"{CURRENT_DIR}/_tmp", exist_ok=True)
    if run_name:
        df.to_excel(
            f"{CURRENT_DIR}/_tmp/info_{datetime_now}_{run_name}_seed{RANDOM_SEED}.xlsx",
            index=False,
        )
    else:
        df.to_excel(
            f"{CURRENT_DIR}/_tmp/info_{datetime_now}_seed{RANDOM_SEED}.xlsx",
            index=False,
        )

    # Plot results
    fig, ax = plt.subplots(1, figsize=(10, 5))
    x = np.arange(vrptw.idx_iteration + 1)
    y1 = np.array(vrptw.global_solution_history) * vrptw.solution_scale_factor
    y2 = np.array(vrptw.fitness_trial_history) * vrptw.solution_scale_factor
    ax.plot(x, y1, marker=".", label="Best Solution")
    ax.plot(x, y2, marker=".", label="Fitness Trial")
    ax.set(
        xlabel="iteration",
        ylabel="Total Distance (Km.)",
        title=f"Reward Total: {reward_total:.2f}, Best Solution: {best_solution:.2f}",
    )
    ax.legend()
    fig.savefig(
        f"{CURRENT_DIR}/_tmp/plot_{datetime_now}_{run_name}_seed{RANDOM_SEED}.png",
        dpi=300,
    )

    # Save the vrptw object
    with open(
        f"{CURRENT_DIR}/_tmp/vrp_{datetime_now}_{run_name}_seed{RANDOM_SEED}.pkl", "wb"
    ) as f:
        pickle.dump(vrptw, f)

    print("------------------------------")
