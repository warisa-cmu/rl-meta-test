# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from P02_MSIE.T01_prototype_class.DE_IM_VRPTW_classV4 import VRPTW


# %%
class AIMH_ENV(gym.Env):
    def __init__(self, vrp, interval_it=100):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([5, 5, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        # order = [
        #     "best_solution",
        #     "F",
        #     "CR",
        #     "MG",
        #     "percent_convergence",
        #     "std_pop",
        #     "count_total_iteration",
        # ]
        self.observation_space = gym.spaces.Box(
            low=np.array([-2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([2, 10, 10, 1, 1, 1, 1e5], dtype=np.float32),
            shape=(7,),  # 7 features
            dtype=np.float64,
        )
        self.vrp = vrp
        self.interval_it = interval_it
        pass

    def _get_obs(self):
        state = self.vrp.get_current_state()
        obs = np.array(
            [
                np.float64(state["best_solution"]),
                np.float64(state["F"]),
                np.float64(state["CR"]),
                np.float64(state["MG"]),
                np.float64(state["percent_convergence"]),
                np.float64(state["std_pop"]),
                np.float64(state["count_total_iteration"]),
            ],
            dtype=np.float32,
        )

        return obs

    def _get_info(self):
        # TODO: We might want to see more stuff here.
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info in addition to the observation
        """
        return {**self.vrp.get_current_state()}

    def reset(self, seed=None, options=None):
        np.random.seed(seed or 42)
        self.vrp.reset()
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.vrp.action(action)
        self.vrp.evolve(n_iteration=self.interval_it)
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


# %%
AIMH_ENV(vrp=None).action_space.sample()

# %%
distance = (
    pd.read_excel(r"src/Source/rl_meta_test_data.xlsx", sheet_name="distance")
    .fillna(9999999)
    .to_numpy()
)

df_vehicle = (
    pd.read_excel(r"src/Source/rl_meta_test_data.xlsx", sheet_name="vehicle")
    .iloc[:, :2]
    .to_numpy(dtype=int)
)
vehicle = df_vehicle[0]

df_101 = pd.read_excel(
    r"src/Source/rl_meta_test_data.xlsx", sheet_name="customer"
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
interval_it = 10
population_size = 4
bounds = np.array([[0, 1]] * dimensions)
F_rate = 0.5
CR_rate = 0.5
MG_rate = 0.5

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
)

# %%
# TODO: Fix the environment to pass the checks
# UserWarning: WARN: The obs returned by the `step()` method is not within the observation space.
#   logger.warn(f"{pre} is not within the observation space.")
# Environment has issues: Deterministic step observations are not equivalent for the same seed and action


from gymnasium.utils.env_checker import check_env

env = AIMH_ENV(vrp=vrptw, interval_it=interval_it)

# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

# %%
from stable_baselines3 import SAC

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)
