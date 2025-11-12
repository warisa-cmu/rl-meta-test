import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC

from P02_MSIE.T05_exp.env_v3 import AIMH_ENV
from P02_MSIE.T05_exp.problem_sets import load_vrp

PROBLEM_SET = "LARGE"
RANDOM_SEED = 42
folder_sets = [
    dict(
        run_name="R1",
        folder="R_20251112_165510",
        file_type="end",
        it=20000,
        attr={"target_solution_unscaled": 250},
    ),
    dict(
        run_name="R2",
        folder="R_20251112_172424",
        file_type="rw",
        it=54469,
        attr={"target_solution_unscaled": 250},
    ),
    dict(
        run_name="R3",
        folder="R_20251112_204554",
        file_type="end",
        it=100000,
        attr={"target_solution_unscaled": 250},
    ),
    dict(
        run_name="R4",
        folder="R_20251112_213633",
        file_type="end",
        it=100000,
        attr={"target_solution_unscaled": 250},
    ),
]

vrptw = load_vrp(problem_set=PROBLEM_SET, verbose=0)

datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
for fs in folder_sets:
    folder = fs["folder"]
    file_type = fs["file_type"]
    it = fs["it"]
    run_name = fs["run_name"]
    print(f"Evaluating folder: {folder}, file_type: {file_type}, it: {it}")
    for attr_key, attr_value in fs["attr"].items():
        if attr_key == "target_solution_unscaled":
            vrptw.set_target_solution(attr_value)
            print(
                "Expected target_solution:",
                attr_value / vrptw.solution_scale_factor,
                "Verifying:",
                getattr(vrptw, "target_solution"),
            )
        else:
            setattr(vrptw, attr_key, attr_value)
            print(
                f"Set vrptw.{attr_key} = {attr_value}",
                "Verifying:",
                getattr(vrptw, attr_key),
            )

    # Load the model
    model = SAC.load(f"save_models/{folder}/{file_type}_{it:05d}")

    # Create the environment
    env = AIMH_ENV(vrp=vrptw)

    # Run the evaluation
    obs, info = env.reset(seed=RANDOM_SEED)
    terminated = False
    truncated = False
    data_array = []
    reward_total = 0
    idx = -1
    while not (terminated or truncated):
        idx += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # if idx % 100 == 0:
        #     print(f"Step: {idx}, Reward: {reward:.2f}, Action: {action}, Info: {info}")
        reward_total += reward
        data_added = {**info, "reward": reward, "action": action}
        data_array.append(data_added)
    df = pd.DataFrame.from_dict(data_array)
    best_solution = df["best_solution"].min() * vrptw.solution_scale_factor

    # Save results
    os.makedirs("_tmp", exist_ok=True)
    if run_name:
        df.to_excel(
            f"_tmp/info_{datetime_now}_{run_name}_seed{RANDOM_SEED}.xlsx", index=False
        )
    else:
        df.to_excel(f"_tmp/info_{datetime_now}_seed{RANDOM_SEED}.xlsx", index=False)

    # Print results
    console = Console()
    table = Table(title="Data")
    for col in df.columns:
        table.add_column(col, style="cyan", no_wrap=True)
    for index, row in df.iterrows():
        table.add_row(*[str(item) for item in row])
    console.print(table)

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
    fig.savefig(f"_tmp/plot_{datetime_now}_{run_name}_seed{RANDOM_SEED}.png", dpi=300)

    # Save the vrptw object
    with open(f"_tmp/vrp_{datetime_now}_{run_name}_seed{RANDOM_SEED}.pkl", "wb") as f:
        pickle.dump(vrptw, f)

    print("------------------------------")
