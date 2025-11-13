import numpy as np
import pandas as pd

from P02_MSIE.T06_exp.vrptw_v9 import VRPTW


def load_vrp(
    problem_set: str,
    verbose: int = 0,
) -> VRPTW:
    # Set hyperparameters based on problem set
    if problem_set == "SMALL":
        PATIENCE = 200
        POPULATION_SIZE = 4
        INTERVAL_IT = 20
        TARGET_SOLUTION_UNSCALED = 48.0
        ALPHA_TARGET = 1
        ALPHA_PATIENCE = 20
        SOLUTION_SCALE_FACTOR = 1
    elif problem_set == "LARGE":
        PATIENCE = 400
        POPULATION_SIZE = 4
        INTERVAL_IT = 10
        TARGET_SOLUTION_UNSCALED = 250
        ALPHA_TARGET = 2
        ALPHA_PATIENCE = 20
        SOLUTION_SCALE_FACTOR = 500
    else:
        raise ValueError("Invalid problem_set. Choose either 'SMALL' or 'LARGE'.")

    # Load data
    if problem_set == "SMALL":
        distance = (
            pd.read_excel(r"./src/Source/rl_meta_test_data.xlsx", sheet_name="distance")
            .fillna(9999999)
            .to_numpy()
        )

        df_vehicle = (
            pd.read_excel(r"./src/Source/rl_meta_test_data.xlsx", sheet_name="vehicle")
            .iloc[:, :2]
            .to_numpy(dtype=int)
        )
        vehicle = df_vehicle[0]

        df_101 = pd.read_excel(
            r"./src/Source/rl_meta_test_data.xlsx", sheet_name="customer"
        ).iloc[:, 3:]

    elif problem_set == "LARGE":
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
    else:
        raise ValueError("Invalid problem_set. Choose either 'SMALL' or 'LARGE'.")

    demand = df_101.iloc[:, 0].to_numpy()
    readyTime = df_101.iloc[:, 1].to_numpy()
    dueDate = df_101.iloc[:, 2].to_numpy()
    serviceTime = df_101.iloc[:, 3].to_numpy()
    dimensions = len(distance) - 1 + vehicle[0]
    bounds = [0, 1]
    vrptw = VRPTW(
        population_size=POPULATION_SIZE,
        dimensions=dimensions,
        bounds=bounds,
        distance=distance,
        demand=demand,
        readyTime=readyTime,
        dueDate=dueDate,
        serviceTime=serviceTime,
        vehicle=vehicle,
        interval_it=INTERVAL_IT,
        patience=PATIENCE,
        solution_scale_factor=SOLUTION_SCALE_FACTOR,
        target_solution_unscaled=TARGET_SOLUTION_UNSCALED,
        alpha_target=ALPHA_TARGET,
        alpha_patience=ALPHA_PATIENCE,
        verbose=verbose,
    )
    return vrptw
