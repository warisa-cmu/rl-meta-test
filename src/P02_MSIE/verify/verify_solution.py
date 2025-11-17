import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------
# VRPTW cost evaluator (your original logic kept intact)
# --------------------------
def preserving_strategy(self, X: np.ndarray, V: np.ndarray) -> float:
    # --- Unpack input data from keyword arguments ---
    dist = self._info["distance"]  # Distance/time matrix between all nodes
    weight = self._info["demand"]  # Demand (weight) for each customer node
    ready = self._info["readyTime"]  # Ready time (earliest service time) for each node
    due = self._info["dueDate"]  # Due time (latest service time) for each node
    service = self._info["serviceTime"]  # Service time at each node
    vehicle = self._info[
        "vehicle"
    ]  # Vehicle info: [number of vehicles, capacity per vehicle]

    # Get per-vehicle capacities (by indexing with V)
    pre_w_cap = np.array([vehicle[1]] * vehicle[0])
    w_cap = pre_w_cap[V]

    # -- Initialization --
    sequence = X  # Route sequence (includes depot at start & end)
    n_cust = len(sequence) - 2  # Number of customers (not counting depot nodes)
    n_veh = vehicle[0] - 1  # Number of vehicles - 1 (for indexing)
    i, k = 0, 0  # i: current position in sequence, k: vehicle index
    total_distance = 0  # Store total traveled distance (with penalty if any)

    # -- Main loop over each vehicle route --
    while k <= n_veh and i <= n_cust:
        # Initialize per-route accumulators
        route_dist, route_time, weight_load, penaltyCost = 0, 0, 0, 0

        if k > 0:
            i += 1  # Move to the next start customer for the next vehicle
        # Start route: depot to first customer
        route_dist += dist[0][sequence[i]]  # Distance depot -> first customer
        route_time += (
            service[0] + dist[0][sequence[i]]
        )  # Service + travel time to first customer
        weight_load += weight[sequence[i]]  # Initial cargo: first customer demand

        if route_time < ready[sequence[i]]:
            route_time = ready[sequence[i]]  # Wait if vehicle arrives before ready time

        if route_time > due[sequence[i]] or weight_load > w_cap[k]:
            penaltyCost += 1e11  # Penalty: arrived after due time (infeasible)
            break

        # --- Continue visiting customers along this route ---
        while i <= n_cust:
            route_dist += dist[sequence[i]][sequence[i + 1]]  # Add next leg distance

            route_time += (
                service[sequence[i]] + dist[sequence[i]][sequence[i + 1]]
            )  # Add service + travel time

            weight_load += weight[sequence[i + 1]]  # Add new customer demand

            if route_time < ready[sequence[i + 1]]:
                route_time = ready[sequence[i + 1]]  # Wait if arrive early at next node

            # If time window or capacity violated, backtrack and finish route
            if route_time > due[sequence[i + 1]] or weight_load > w_cap[k]:
                route_dist -= dist[sequence[i]][sequence[i + 1]]
                route_time -= service[sequence[i]] + dist[sequence[i]][sequence[i + 1]]
                weight_load -= weight[sequence[i + 1]]
                break
            i += 1

        # --- Finish by returning to depot ---
        route_dist += dist[sequence[i]][0]  # Add distance to depot
        route_time += (
            service[sequence[i]] + dist[sequence[i]][0]
        )  # Add service at last node + travel to depot
        if route_time > due[0]:
            penaltyCost += 1e11  # Penalty: returned to depot too late
        # Accumulate this route's total (distance + penalty if any)
        total_distance += route_dist + penaltyCost
        k += 1  # Next vehicle

    return (
        total_distance  # Return overall objective (distance with penalty if violated)
    )


# --------------------------
# Decode a chromosome to (sequence, vehicle) and evaluate
# --------------------------
def objective_func(self, population: np.ndarray) -> float:
    vehicle = self._info["vehicle"]
    # First block → customer order (rank-based decoding): smaller value → earlier visit
    seq = population[: -vehicle[0]].argsort() + 1
    # Last 'num_vehicles' genes → vehicle order/index (again rank-based)
    sort = population[-vehicle[0] :].argsort()
    obj_val = self.preserving_strategy(seq, sort)
    return obj_val


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


demand = df_101.iloc[:, 0].to_numpy()
readyTime = df_101.iloc[:, 1].to_numpy()
dueDate = df_101.iloc[:, 2].to_numpy()
serviceTime = df_101.iloc[:, 3].to_numpy()

pos = np.array(
    [
        0.50369861,
        0.08719329,
        0.01180275,
        0.08087462,
        0.70414424,
        0.86386483,
        0.73380336,
        0.733839,
        0.79324075,
        0.76408076,
        0.78350367,
        0.48413228,
        0.12843212,
        0.48282761,
        0.46392457,
        0.47033888,
        0.24535577,
        0.35689161,
        0.39245587,
        0.55761789,
        0.67366081,
        0.64257695,
        0.62299986,
        0.60773028,
        0.62008494,
        0.58018575,
        0.6419171,
        0.82608662,
        0.84369533,
        0.94276053,
        0.26714116,
        0.68538872,
        0.63840237,
        0.39217874,
        1.0,
        0.76462787,
        0.68375802,
        0.46648868,
        0.2903476,
        0.37217975,
        0.32478988,
        0.28373446,
        0.56615504,
        0.47393336,
        0.15465599,
        0.0,
        0.487033,
        0.4587994,
        0.54145525,
        0.59178383,
    ]
)
print(pos)
