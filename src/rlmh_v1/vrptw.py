import time
from dataclasses import dataclass, field, fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from P02_MSIE.T11_refactor.utils import LinearScaler, RewardParams


@dataclass
class VRPTW:
    # Differential Evolution (DE) optimizer wrapper for VRPTW.
    # - Genome is real-valued; the first block decodes to a customer order (via argsort),
    #   the last 'num_vehicles' genes decode to a vehicle ordering/indexing.
    # - Fitness is computed by 'preserving_strategy' (your VRPTW cost/feasibility evaluator).
    # ----- Inputs -----
    population_size: int  # Population size for DE
    dimensions: int  # Chromosome length
    bounds: tuple  # Gene bounds
    # Problem info passed into objective/preserving_strategy
    distance: np.ndarray
    demand: np.ndarray
    readyTime: np.ndarray
    dueDate: np.ndarray
    serviceTime: np.ndarray
    vehicle: np.ndarray
    # RL
    interval_it: int  # Interval iteration between action
    patience: int  # Patience for early stopping
    target_solution: float  # Target solution (unscaled)
    sc_solution: LinearScaler  # Scaler for solution cost
    sc_iteration: LinearScaler  # Scaler for iteration
    sc_F: LinearScaler  # Scaler for mutation factor F
    sc_CR: LinearScaler  # Scaler for crossover rate CR
    sc_MG: LinearScaler  # Scaler for migration rate MG
    reward_params: RewardParams  # Reward parameters
    verbose: int = 0  # Verbosity level
    convert_none_seed_to_number: bool = (
        True  # During training, seed=None is passed; convert to 42 for reproducibility
    )
    seed: int | None = None  # Random seed used

    # ----- Internals -----
    _info: dict = field(default_factory=dict)  # Internal storage for problem info
    F_rate: float = 0  # DE mutation factor
    CR_rate: float = 0  # DE crossover rate
    MG_rate: float = 0  # migration rate (fraction of elites)
    global_solution_history: list = field(
        default_factory=list
    )  # Best fitness per iteration
    fitness_trial_history: list = field(
        default_factory=list
    )  # Fitness of trial solutions
    current_cost: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Current cost values
    current_fitness_trials: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Current fitness trial values
    population: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Current population
    idx_iteration: int = -1  # Current iteration index
    local_rng: np.random.Generator = (
        np.random.default_rng()
    )  # Local random number generator
    patience_remaining: int = 0  # Remaining patience

    def __post_init__(self):
        # Problem info passed into objective/preserving_strategy
        self._info = {
            "distance": self.distance,
            "demand": self.demand,
            "readyTime": self.readyTime,
            "dueDate": self.dueDate,
            "serviceTime": self.serviceTime,
            "vehicle": self.vehicle,
        }

        # Check bounds shape
        if self.bounds[0] != 0 and self.bounds[1] != 1:
            raise ValueError(
                "Bounds should be [0, 1] for min-max scaling. If different, please adjust the code accordingly."
            )

        # Population has to be larger than 4 due to random.choice
        if self.population_size < 4:
            raise ValueError("Population size must be at least 4.")

    # --------------------------
    # Initialize population and DE params
    # --------------------------
    def reset(self, seed=42):
        # I found that SB3 passed None as seed and the value will not get set properly.
        if seed is None and self.convert_none_seed_to_number:
            seed = 42
        if self.verbose > 0:
            print(f"Seed used in reset: {seed}")
        self.seed = seed

        # Initialize RNG
        self.local_rng = np.random.default_rng(seed)

        # Random uniform initialization within per-gene bounds
        self.population = self.local_rng.uniform(
            self.bounds[0],
            self.bounds[1],
            (self.population_size, self.dimensions),
        )

        self.current_cost = np.ones(shape=(self.population_size,))
        self.current_fitness_trials = np.ones(shape=(self.population_size,))
        for i in range(self.population_size):
            objective_value = self.objective_func(self.population[i])
            self.current_cost[i] = objective_value
            self.current_fitness_trials[i] = objective_value

        # Reasonable default rates
        self.F_rate = self.sc_F.get_starting_value(scaled=False)
        self.CR_rate = self.sc_CR.get_starting_value(scaled=False)
        self.MG_rate = self.sc_MG.get_starting_value(scaled=False)
        self.idx_iteration = -1  # This is to offset the first increment in evolve()
        self.global_solution_history = []
        self.fitness_trial_history = []
        self.patience_remaining = self.patience

    # --------------------------
    # VRPTW cost evaluator (your original logic kept intact)
    # --------------------------
    def preserving_strategy(self, X: np.ndarray, V: np.ndarray) -> float:
        # --- Unpack input data from keyword arguments ---
        dist = self._info["distance"]  # Distance/time matrix between all nodes
        weight = self._info["demand"]  # Demand (weight) for each customer node
        ready = self._info[
            "readyTime"
        ]  # Ready time (earliest service time) for each node
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
                route_time = ready[
                    sequence[i]
                ]  # Wait if vehicle arrives before ready time

            if route_time > due[sequence[i]] or weight_load > w_cap[k]:
                penaltyCost += 1e11  # Penalty: arrived after due time (infeasible)
                break

            # --- Continue visiting customers along this route ---
            while i <= n_cust:
                route_dist += dist[sequence[i]][
                    sequence[i + 1]
                ]  # Add next leg distance

                route_time += (
                    service[sequence[i]] + dist[sequence[i]][sequence[i + 1]]
                )  # Add service + travel time

                weight_load += weight[sequence[i + 1]]  # Add new customer demand

                if route_time < ready[sequence[i + 1]]:
                    route_time = ready[
                        sequence[i + 1]
                    ]  # Wait if arrive early at next node

                # If time window or capacity violated, backtrack and finish route
                if route_time > due[sequence[i + 1]] or weight_load > w_cap[k]:
                    route_dist -= dist[sequence[i]][sequence[i + 1]]
                    route_time -= (
                        service[sequence[i]] + dist[sequence[i]][sequence[i + 1]]
                    )
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

        return total_distance  # Return overall objective (distance with penalty if violated)

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

    # --------------------------
    # Main DE loop
    # --------------------------
    def evolve(self):
        for _ in range(self.interval_it):
            self.idx_iteration += 1
            for i in range(self.population_size):
                # ---- Mutation (DE/rand/1) ----
                indices = [idx for idx in range(self.population_size) if idx != i]
                b, c = self.population[self.local_rng.choice(indices, 2, replace=False)]
                mutant = self.population[i] + self.F_rate * (b - c)

                # (Optional) You might want to clip mutant to bounds to avoid runaway values:
                # mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                mutant = self.minmax_scaling(mutant)

                # ---- Crossover (binomial) ----
                crossover_prob = self.local_rng.random(self.dimensions)
                trial = np.where(
                    crossover_prob < self.CR_rate, mutant, np.copy(self.population[i])
                )

                # Note: In classic DE, we ensure at least one dimension from the mutant (jrand trick).
                # Todo: enforce jrand if you want strict DE semantics.

                # ---- Selection ----
                fitness_trial = self.objective_func(trial)
                self.current_fitness_trials[i] = fitness_trial
                fitness_current = self.current_cost[i]

                if fitness_trial < fitness_current:
                    self.population[i] = trial
                    self.current_cost[i] = fitness_trial

            # Store best solution of this iteration
            self.global_solution_history.append(self.calc_best_solution(type="cost"))
            # Store best trial fitness of this iteration
            self.fitness_trial_history.append(self.calc_best_solution(type="trial"))

            # Update patience
            if len(self.global_solution_history) > 1:
                diff_amount = (
                    self.global_solution_history[-2] - self.global_solution_history[-1]
                )
                if self.verbose > 0:
                    print(
                        f"Iteration {self.idx_iteration}: Best Solution = {self.global_solution_history[-1]}, Diff = {diff_amount}, Patience Remaining = {self.patience_remaining}"
                    )
                if diff_amount > 0:  # Improvement found
                    self.patience_remaining = self.patience  # Reset patience
                elif diff_amount == 0:  # No improvement
                    self.patience_remaining -= 1
                else:  # Should not happen
                    raise Exception("Best solution worsened during evolution!")
            pass

    def migration(self):
        # Island Model Migration

        # Select elites from current population
        num_elites = int(self.population_size * self.MG_rate)
        if num_elites == 0:
            # Ensure at least one elite is selected
            num_elites = 1
        elif num_elites >= self.population_size:
            return  # No migration needed if all are elites

        idx_elites = self.current_cost.argsort()[:num_elites]
        population_elites = self.population[idx_elites]
        cost_elites = self.current_cost[idx_elites]

        # New random population
        population_migrate = self.local_rng.uniform(
            self.bounds[0],
            self.bounds[1],
            (self.population_size, self.dimensions),
        )

        # Evaluate new population
        cost_migrate = np.zeros(shape=(self.population_size,))
        for obj in range(self.population_size):
            cost_migrate[obj] = self.objective_func(population_migrate[obj])

        # Replace the worst in the new population with previous elites
        idx_bad_migration = cost_migrate.argsort()[::-1][:num_elites]
        population_migrate[idx_bad_migration] = population_elites
        cost_migrate[idx_bad_migration] = cost_elites

        # Adopt the migrated population
        self.population = population_migrate
        self.current_cost = cost_migrate

        if len(self.global_solution_history) > 0:
            solution_before = self.global_solution_history[-1]
            solution_after = self.calc_best_solution(type="cost")
            # Check if best solution improved
            if solution_after < solution_before:
                self.patience_remaining = self.patience  # Reset patience
                if self.verbose > 0:
                    print(
                        f"Migration improved best solution from {solution_before} to {solution_after}"
                    )
            # Check that best solution did not worsen
            elif solution_after > solution_before:
                raise Exception("Best solution worsened after migration!")
        pass

    def action(self, option):
        # option: [F_rate, CR_rate, MG_rate] (scaled)
        self.F_rate = self.sc_F.inverse_transform(option[0])
        self.CR_rate = self.sc_CR.inverse_transform(option[1])
        self.MG_rate = self.sc_MG.inverse_transform(option[2])

        # Clamp F and CR to their bounds
        self.F_rate = self.sc_F.clip_to_bounds(self.F_rate, scaled=False)
        self.CR_rate = self.sc_CR.clip_to_bounds(self.CR_rate, scaled=False)
        self.MG_rate = self.sc_MG.clip_to_bounds(self.MG_rate, scaled=False)
        # Note that migration can skip if MG_rate is too low.
        self.migration()

    def calc_robustness(self, solution):
        # Return the std of current generation's fitness values.
        # Note: call 'evolve()' at least once so 'current_cost' is populated.
        # self.DE_robust = np.std(solution)
        # return self.DE_robust
        raise NotImplementedError("calc_robustness is not implemented yet.")

    def calc_std(self, type="cost", scaled=False):
        if type == "cost":
            solution_arrays = self.current_cost
        elif type == "trial":
            solution_arrays = self.current_fitness_trials
        else:
            raise Exception("Invalid Option")

        if not scaled:
            return np.std(solution_arrays)
        else:
            return np.std(self.sc_solution.transform(solution_arrays))

    def calc_convergence_rate(self):
        if len(self.global_solution_history) < self.interval_it:
            return 0  # Not enough history yet

        start_it = self.idx_iteration - self.interval_it + 1
        val_start = self.global_solution_history[start_it]
        val_end = self.global_solution_history[-1]
        convergence_rate = np.abs(
            (val_end - self.target_solution) / (val_start - self.target_solution)
        )
        self.convergence_rate = convergence_rate
        return convergence_rate

    # --------------------------
    # Introspection helpers
    # --------------------------
    def get_current_state(self):
        return {
            "F_sc": self.sc_F.transform(self.F_rate),
            "CR_sc": self.sc_CR.transform(self.CR_rate),
            "MG_sc": self.sc_MG.transform(self.MG_rate),
            "best_solution_sc": self.sc_solution.transform(
                self.calc_best_solution(type="cost")
            ),
            "convergence_rate": self.calc_convergence_rate(),  # No scaling needed
            "std_population_sc": self.calc_std(type="cost", scaled=True),
            "best_trial_fitness_sc": self.sc_solution.transform(
                self.calc_best_solution(type="trial")
            ),
            "std_trial_fitness_sc": self.calc_std(type="trial", scaled=True),
            "patience_remaining_sc": self.patience_remaining / self.patience,
        }

    def get_info(self):
        state_scaled = self.get_current_state()
        state = {
            "F": self.F_rate,
            "CR": self.CR_rate,
            "MG": self.MG_rate,
            "best_solution": self.calc_best_solution(type="cost"),
            "convergence_rate": self.calc_convergence_rate(),
            "std_population": self.calc_std(type="cost", scaled=False),
            "total_iteration": self.idx_iteration + 1,
            "best_trial_fitness": self.calc_best_solution(type="trial"),
            "std_trial_fitness": self.calc_std(type="trial", scaled=False),
            "patience_remaining": self.patience_remaining,
        }

        return {
            "idx_iteration": self.idx_iteration,
            **state,
            **state_scaled,
        }

    def calc_best_solution(self, type="cost"):
        # Evaluate and return the best fitness among current population
        if type == "cost":
            obj_values = self.current_cost
        elif type == "trial":
            obj_values = self.current_fitness_trials
        else:
            raise Exception("Invalid Option")

        # Find the index of the individual with the lowest (best) objective value.
        best_index = np.argmin(obj_values)

        # Get the best objective value and the corresponding individual (solution).
        best_solution = obj_values[best_index]

        return best_solution

    def is_terminated(self):
        # Check termination conditions based on max iterations or patience.
        if self.idx_iteration >= self.sc_iteration.bounds[1]:
            return True
        else:
            if self.patience_remaining <= 0:
                return True
            else:
                return False

    def get_reward(self):
        mode = self.reward_params.reward_mode
        # Take care of the special case where there is not enough history yet
        if len(self.global_solution_history) < 2 or len(self.fitness_trial_history) < 2:
            return 0

        # Calculate start and end iteration indices
        start_it = self.idx_iteration - self.interval_it + 1
        end_it = self.idx_iteration + 1
        # Handle edge case where start_it is negative
        if start_it < 0:
            return 0
        # Handle edge case where start_it is 0 and shifting the index (-1) will not work
        if start_it == 0:
            start_it = 1

        # Compute reward based on the selected mode
        if mode == "TARGET_SIMPLE":
            # Simple reward based on distance to target solution
            return self.sc_solution.transform(self.global_solution_history[-1])
        elif mode == "CUMULATIVE_DIFF":
            # Calculate average reward by comparing best solution history and trial fitness history (shifted by 1)
            best_solution = self.global_solution_history[start_it - 1 : end_it - 1]
            trial_solution = self.fitness_trial_history[start_it:end_it]
            best_solution_sc = self.sc_solution.transform(
                best_solution, keep_list=False
            )  # np array
            trial_solution_sc = self.sc_solution.transform(
                trial_solution, keep_list=False
            )  # np array
            reward_ave = np.sum(best_solution_sc - trial_solution_sc) / self.interval_it
            return reward_ave
        elif mode in [
            "TARGET_ENHANCED_1",
            "TARGET_ENHANCED_2",
            "TARGET_ENHANCED_3",
        ]:
            epsilon_target = 1e-6  # Prevent division by zero

            # Calculate improvement over the interval
            improvement = (
                self.global_solution_history[start_it]
                - self.global_solution_history[self.idx_iteration]
            )
            improvement_sc = self.sc_solution.transform(improvement)

            # Get current best solution
            value = self.global_solution_history[-1]
            value_sc = self.sc_solution.transform(value)

            # Get target solution (scaled)
            target_solution_sc = self.sc_solution.transform(self.target_solution)

            # Calculate closeness to target solution
            alpha_target = self.reward_params.alpha_target
            if mode == "TARGET_ENHANCED_2":
                # Inverse difference to target
                close_to_target = 1 / (
                    np.abs(value_sc - target_solution_sc) + epsilon_target
                )
                # Calculate final reward
                reward_1 = improvement_sc + alpha_target * close_to_target
            elif mode == "TARGET_ENHANCED_3":
                # Huber-like soft distance to target
                s = self.reward_params.s
                c = self.reward_params.c
                d = np.maximum(value_sc - target_solution_sc, 0.0)
                close_to_target = 1 / (1 + np.power((d / c), s))
                reward_1 = improvement_sc + alpha_target * close_to_target
            if self.verbose > 0:
                print(
                    f"Improvement: {improvement_sc}, Close to target: {close_to_target * self.alpha_target}, Reward1: {reward_1}"
                )

            if mode == "TARGET_ENHANCED_1":
                return reward_1
            elif mode in ["TARGET_ENHANCED_2", "TARGET_ENHANCED_3"]:
                # Reset patience if there is an improvement between start_it and current
                if improvement_sc > 0:
                    self.patience_remaining = self.patience  # Reset patience

                # Scaled by patience
                alpha_patience = self.reward_params.alpha_patience
                p = float(np.clip(self.patience_remaining / self.patience, 0.0, 1.0))
                patience_factor = max(0, np.exp(-alpha_patience * (1.0 - p)))

                improvement_bonus = 0
                if mode == "TARGET_ENHANCED_2":
                    reward_2 = reward_1 * patience_factor
                elif mode == "TARGET_ENHANCED_3":
                    # improvement_bonus = 0.5 if improvement_sc > 0 else 0
                    reward_2 = reward_1 * patience_factor + improvement_bonus
                if self.verbose > 0:
                    print(
                        f"Improvement: {improvement_sc}, Improvement Bonus: {improvement_bonus}, Close to target: {close_to_target * self.alpha_target}, Reward before: {reward_1}, p_factor: {patience_factor}, Reward after: {reward_2}"
                    )
                return reward_2
        else:
            raise Exception("Invalid Option")

    def minmax_scaling(self, arr):
        # Find the minimum and maximum values in the array
        min_val = np.min(arr)
        max_val = np.max(arr)

        # Perform min-max normalization
        # Ensure to handle the case where max_val - min_val is zero to avoid division by zero
        if max_val - min_val == 0:
            normalized_array = np.zeros_like(
                arr, dtype=float
            )  # All values become 0 if range is 0
        else:
            normalized_array = (arr - min_val) / (max_val - min_val)
        return normalized_array


@dataclass
class VRPTW_INPUT_PARAMS:
    problem_set: str
    population_size: int


@dataclass
class RL_INPUT_PARAMS:
    sc_F: LinearScaler
    sc_CR: LinearScaler
    sc_MG: LinearScaler
    sc_solution: LinearScaler
    sc_iteration: LinearScaler
    interval_it: int
    target_solution: float
    reward_params: RewardParams
    patience: int
    verbose: int
    convert_none_seed_to_number: bool


def load_vrptw(
    vrptw_input_params: VRPTW_INPUT_PARAMS, rl_input_params: RL_INPUT_PARAMS
):
    if vrptw_input_params.problem_set == "SMALL":
        excel_file = "./src/Source/rl_meta_test_data.xlsx"
    elif vrptw_input_params.problem_set == "LARGE":
        excel_file = "./src/Source/rl_meta_test_data_25_customer.xlsx"
    elif vrptw_input_params.problem_set == "LARGE2":
        excel_file = "./src/Source/r101_25customers_datasets.xlsx"
    elif vrptw_input_params.problem_set == "LARGE3":
        excel_file = "./src/Source/rc101_25customers_datasets.xlsx"
    else:
        raise ValueError("Invalid problem_set. Choose either 'SMALL' or 'LARGE'.")

    # Load distance data
    df_distance = pd.read_excel(excel_file, sheet_name="distance")
    distance = df_distance.fillna(9999999).to_numpy()
    # Load vehicle data
    df_vehicle = pd.read_excel(excel_file, sheet_name="vehicle")
    vehicle = df_vehicle.loc[0, "fleet_size":"fleet_capacity"].values
    # Load customer data
    df_customer = pd.read_excel(excel_file, sheet_name="customer")
    demand = df_customer.loc[:, "demand"].to_numpy()
    readyTime = df_customer.loc[:, "readyTime"].to_numpy()
    dueDate = df_customer.loc[:, "dueTime"].to_numpy()
    serviceTime = df_customer.loc[:, "duration"].to_numpy()
    dimensions = distance.shape[0] - 1 + vehicle[0]
    #
    bounds = (0, 1)

    _info_vrptw = {
        "distance": distance,
        "demand": demand,
        "readyTime": readyTime,
        "dueDate": dueDate,
        "serviceTime": serviceTime,
        "vehicle": vehicle,
        "population_size": vrptw_input_params.population_size,
        "dimensions": dimensions,
        "bounds": bounds,
    }
    #
    # Cannot as asdict directly due to possible nested dataclasses
    # _info_RL = asdict(params_RL) # This does not work for nested dataclasses
    _info_RL = {
        f.name: getattr(rl_input_params, f.name) for f in fields(rl_input_params)
    }
    #
    vrptw = VRPTW(**_info_vrptw, **_info_RL)
    return vrptw


if __name__ == "__main__":
    vpr_input_params = VRPTW_INPUT_PARAMS(
        problem_set="SMALL",  # Options: "SMALL", "LARGE"
        population_size=40,
    )
    PATIENCE = 200
    VERBOSE = 0

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
            convert_none_seed_to_number=True,
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
            convert_none_seed_to_number=True,
        )

    vrptw = load_vrptw(vpr_input_params, rl_input_params)

    vrptw.reset()
    start = time.time()
    idx = -1
    while not vrptw.is_terminated():
        idx += 1
        vrptw.evolve()
        vrptw.migration()
        reward = vrptw.get_reward()
        if idx % 100 == 0:
            best_solution = (
                vrptw.global_solution_history[-1]
                if len(vrptw.global_solution_history) > 0
                else "N/A"
            )
            print(
                f"Idx: {idx}, Best Solution: {best_solution:4.3f}, Reward: {reward:4.3f}, F: {vrptw.F_rate:4.3f}, CR: {vrptw.CR_rate:4.3f}, MG: {vrptw.MG_rate:4.3f}, Patience Remaining: {vrptw.patience_remaining}"
            )

    end = time.time()
    computational_time = end - start
    print(f"Computational time : {computational_time} second")
    fig, ax = plt.subplots(1, figsize=(10, 5))
    x = np.arange(vrptw.idx_iteration + 1)
    y1 = vrptw.global_solution_history
    y2 = vrptw.fitness_trial_history
    ax.plot(x, y1, marker=".", label="Best Solution")
    ax.plot(x, y2, marker=".", label="Fitness Trial")
    ax.set(
        xlabel="iteration",
        ylabel="Total Distance (Km.)",
        title="Differential Evoluation + Island Model Algorithm Replication",
    )
    ax.legend()
    plt.show()
