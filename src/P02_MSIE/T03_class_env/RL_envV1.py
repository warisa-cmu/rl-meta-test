from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from P02_MSIE.T03_class_env.DE_IM_VRPTW_classV6 import (
    CR_LOWER_BOUND,
    CR_UPPER_BOUND,
    F_LOWER_BOUND,
    F_UPPER_BOUND,
    MAX_ITERATION,
    SOL_UPPER_BOUND,
    VAL_INVALID_STD_POPULATION,
    VRPTW,
)


class AIMH_ENV(gym.Env):
    def __init__(self, vrp):
        super().__init__()

        # Action space: [F, CR, Mutation Rate]
        self.action_space = spaces.Box(
            low=np.array([F_LOWER_BOUND, CR_LOWER_BOUND, 0], dtype=np.float32),
            high=np.array([F_UPPER_BOUND, CR_UPPER_BOUND, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        # Observation space features:
        # order = [
        #     "best_solution",
        #     "F",
        #     "CR",
        #     "MG",
        #     "convergence_rate",
        #     "std_pop",
        #     "total_iteration",
        # ]
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    0,  # best_solution lower bound
                    F_LOWER_BOUND,
                    CR_LOWER_BOUND,
                    0,  # MG lower bound
                    0,  # Convergence rate lower bound
                    0,  # std_pop lower bound
                    0,  # total_iteration lower bound
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    SOL_UPPER_BOUND,
                    F_UPPER_BOUND,
                    CR_UPPER_BOUND,
                    1,  # MG upper bound
                    1,  # Convergence_rate upper bound
                    VAL_INVALID_STD_POPULATION,  # std_pop upper bound
                    MAX_ITERATION,
                ],
                dtype=np.float32,
            ),
            shape=(7,),  # 7 features
            dtype=np.float32,
        )
        self.vrp = vrp
        pass

    def _get_obs(self):
        state = self.vrp.get_current_state()
        obs = np.array(
            [
                state["best_solution"],
                state["F"],
                state["CR"],
                state["MG"],
                state["convergence_rate"],
                state["std_pop"],
                state["total_iteration"],
            ],
            dtype=np.float32,
        )

        return obs

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info in addition to the observation
        """
        return self.vrp.get_info()

    def reset(self, seed=None, options=None):
        # np.random.seed(seed or 42)
        self.vrp.reset()
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.vrp.action(action)
        self.vrp.evolve()
        reward = self.vrp.get_reward()
        if self.vrp.is_terminated():
            terminated = True
            truncated = True
        else:
            terminated = False
            truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info


class CustomLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_frequency = 2  # Log every n steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_frequency == 0:
            # Access internal model state and log desired values
            # For example, log the current learning rate:
            # self.logger.record(
            #     "train/learning_rate", self.model.lr_schedule(self.progress_remaining)
            # )

            # You can also access information from the environment or other sources
            # self.logger.record("custom_metric/my_value", some_calculated_value)

            # Access current observation
            current_obs = self.locals["new_obs"]
            self.logger.record("env/current_observation", current_obs)

            # Access current reward
            current_reward = self.locals["rewards"]
            self.logger.record("env/current_reward", current_reward)

            # Access done status
            is_done = self.locals["dones"]
            self.logger.record("env/is_done", is_done)

            # Access info dictionary
            info_dict = self.locals["infos"]
            self.logger.record("env/info_dict", str(info_dict))

            # TODO: Not working yet - log mean episode reward
            if "rollout/ep_rew_mean" in self.logger.name_to_value:
                mean_reward = self.logger.name_to_value["rollout/ep_rew_mean"]
                self.logger.record("env/mean_reward", mean_reward)

            self.logger.dump(self.num_timesteps)
        return True  # Return True to continue training


if __name__ == "__main__":
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
    demand = df_101.iloc[:, 0].to_numpy()
    readyTime = df_101.iloc[:, 1].to_numpy()
    dueDate = df_101.iloc[:, 2].to_numpy()
    serviceTime = df_101.iloc[:, -1].to_numpy()

    kwargs = {
        "distance": distance,
        "demand": demand,
        "readyTime": readyTime,
        "dueDate": dueDate,
        "serviceTime": serviceTime,
        "vehicle": vehicle,
    }
    dimensions = len(distance) - 1 + vehicle[0]
    interval_it = 20
    patience = 100
    population_size = 4
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
        target_solution=40,
    )

    log_path = "./logs"
    date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = AIMH_ENV(vrp=vrptw)
    env = Monitor(
        env,
        filename=f"{log_path}/{date_prefix}_monitor.csv",
    )

    # This will catch many common issues
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
    model = SAC("MlpPolicy", env, verbose=1)

    # Set up custom logger
    new_logger = configure(log_path, format_strings=["stdout", "csv"])
    model.set_logger(new_logger)
    # model.learn(total_timesteps=10, callback=CustomLoggerCallback())
    model.learn(total_timesteps=1000)

    obs, info = env.reset()
    terminated = False
    truncated = False
    data_array = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        data_added = {**info, "reward": reward, "action": action}
        data_array.append(data_added)
    df = pd.DataFrame.from_dict(data_array)

    # df.to_excel(f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", index=False)

    console = Console()
    table = Table(title="Data")
    for col in df.columns:
        table.add_column(col, style="cyan", no_wrap=True)
    for index, row in df.iterrows():
        table.add_row(*[str(item) for item in row])
    console.print(table)

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
