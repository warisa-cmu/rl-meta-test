import numpy as np
import pandas as pd

from P02_MSIE.T04_class_env.DE_IM_VRPTW_classV7 import VRPTW


def load_vrp(
    PROBLEM_SET: str,
    patience: int,
    population_size: int,
    interval_it: int,
    target_solution: float,
) -> VRPTW:
    if PROBLEM_SET == "SMALL":
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

    elif PROBLEM_SET == "LARGE":
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
        raise ValueError("Invalid PROBLEM_SET. Choose either 'SMALL' or 'LARGE'.")

    demand = df_101.iloc[:, 0].to_numpy()
    readyTime = df_101.iloc[:, 1].to_numpy()
    dueDate = df_101.iloc[:, 2].to_numpy()
    serviceTime = df_101.iloc[:, 3].to_numpy()
    dimensions = len(distance) - 1 + vehicle[0]
    bounds = np.array([[0, 1]] * dimensions)
    vrptw = VRPTW(
        population_size=population_size,
        dimensions=dimensions,
        bounds=bounds,
        distance=distance,
        demand=demand,
        readyTime=readyTime,
        dueDate=dueDate,
        serviceTime=serviceTime,
        vehicle=vehicle,
        interval_it=interval_it,
        patience=patience,
        target_solution=target_solution,
    )
    return vrptw
