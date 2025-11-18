import os
import pathlib
import pickle
import time
from datetime import datetime
from dataclasses import dataclass, field

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
from stable_baselines3.common.logger import CSVOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor

from P03_MSIE.T06_large_3_v3.utils import LinearScaler, RewardParams
from P03_MSIE.T06_large_3_v3.vrptw import (
    VRPTW,
    VRPTW_INPUT_PARAMS,
    RL_INPUT_PARAMS,
    load_vrptw,
)


class RLMH_ENV(gym.Env):
    metadata = {"render_modes": None, "render_fps": 4}

    def __init__(self, vrp: VRPTW):
        super().__init__()

        # Action space: [F, CR, Mutation Rate] (Scaled)
        self.action_space = spaces.Box(
            low=np.array(
                [
                    vrp.sc_F.bounds_scaled[0],
                    vrp.sc_CR.bounds_scaled[0],
                    vrp.sc_MG.bounds_scaled[0],
                ],
                dtype=np.float64,
            ),
            high=np.array(
                [
                    vrp.sc_F.bounds_scaled[1],
                    vrp.sc_CR.bounds_scaled[1],
                    vrp.sc_MG.bounds_scaled[1],
                ],
                dtype=np.float64,
            ),
            shape=(3,),
            dtype=np.float64,
        )
        # Observation space features:
        # order = [
        #     "F_sc",
        #     "CR_sc",
        #     "MG_sc",
        #     "best_solution_sc",
        #     "convergence_rate", # Already scaled [0,1]
        #     "std_population_sc",
        #     "best_trial_fitness_sc",
        #     "std_trial_fitness_sc",
        #     "patience_remaining_sc"
        # ]
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    vrp.sc_F.bounds_scaled[0],
                    vrp.sc_CR.bounds_scaled[0],
                    vrp.sc_MG.bounds_scaled[0],
                    vrp.sc_solution.bounds_scaled[0],
                    0,  # Convergence rate lower bound
                    0,  # std_population_sc lower bound
                    vrp.sc_solution.bounds_scaled[
                        0
                    ],  # best_trial_fitness_sc lower bound
                    0,  # std_trial_fitness_sc lower bound
                    0,  # patience_remaining_sc lower bound
                ],
                dtype=np.float64,
            ),
            high=np.array(
                [
                    vrp.sc_F.bounds_scaled[1],
                    vrp.sc_CR.bounds_scaled[1],
                    vrp.sc_MG.bounds_scaled[1],
                    vrp.sc_solution.bounds_scaled[1],
                    1,  # Convergence_rate upper bound
                    10,  # std_population_sc upper bound
                    vrp.sc_solution.bounds_scaled[
                        1
                    ],  # best_trial_fitness_sc upper bound
                    vrp.sc_solution.bounds_scaled[
                        1
                    ],  # std_trial_fitness_sc upper bound
                    1,  # patience_remaining_sc upper bound
                ],
                dtype=np.float64,
            ),
            shape=(9,),  # 9 features
            dtype=np.float64,
        )
        self.vrp = vrp
        self.verbose = 0

    def _get_obs(self):
        state = self.vrp.get_current_state()
        obs = np.array(
            [
                state["F_sc"],
                state["CR_sc"],
                state["MG_sc"],
                state["best_solution_sc"],
                state["convergence_rate"],
                state["std_population_sc"],
                state["best_trial_fitness_sc"],
                state["std_trial_fitness_sc"],
                state["patience_remaining_sc"],
            ],
            dtype=np.float64,
        )
        return obs

    def _get_info(self):
        return self.vrp.get_info()

    def reset(self, seed=None, options=None):
        if self.verbose > 0:
            print(f"Environment reset with seed: {seed}")
        super().reset(seed=seed)
        self.vrp.reset(seed=seed)
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


class CustomCallback(BaseCallback):
    def __init__(
        self,
        check_freq: int,
        save_dir: str,
        date_prefix: str,
        verbose=1,
        save_best_reward=False,
        save_best_solution=True,
        save_interval=True,
        save_interval_seconds: int = 8 * 60,  # 8 minutes
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.date_prefix = (
            date_prefix if date_prefix else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.save_dir = f"{save_dir}/R_{self.date_prefix}"
        self.best_global_value = np.inf
        self.best_reward = -np.inf
        self.experiences = []
        self.save_best_reward = save_best_reward
        self.save_best_solution = save_best_solution
        self.save_interval = save_interval
        self.save_interval_seconds = save_interval_seconds
        self._last_save = None

    def _init_callback(self):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self):
        self._last_save = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save experiences
            self.save_experiences()

            # Check if the episode is done
            if self.locals["dones"][0]:
                infos = self.locals["infos"][0]
                if "episode" not in infos.keys() or "best_solution" not in infos.keys():
                    print("No episode or best_solution info found.")
                    return
                best_solution = infos["best_solution"]
                episode_reward = infos["episode"]["r"]
                episode_length = infos["episode"]["l"]
                now = time.time()

                if best_solution < self.best_global_value:
                    self.best_global_value = best_solution
                    if self.verbose > 0:
                        print(
                            f"New best global value: {self.best_global_value:5.3f} at step {self.num_timesteps}",
                        )
                        self.print_info(
                            best_solution,
                            self.best_global_value,
                            episode_reward,
                            episode_length,
                        )
                    if self.save_best_solution:
                        self.save_model(mode="solution")
                elif episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    if self.verbose > 0:
                        print(
                            f"New best reward: {self.best_reward:5.3f} at step {self.num_timesteps}",
                        )
                        self.print_info(
                            best_solution,
                            self.best_global_value,
                            episode_reward,
                            episode_length,
                        )
                    if self.save_best_reward:
                        self.save_model(mode="reward")
                elif now - self._last_save >= self.save_interval_seconds:
                    self._last_save = now
                    if self.verbose > 0:
                        print(
                            f"Interval save at step {self.num_timesteps}",
                        )
                        self.print_info(
                            best_solution,
                            self.best_global_value,
                            episode_reward,
                            episode_length,
                        )
                    if self.save_interval:
                        self.save_model(mode="interval")

                # Delete experiences
                self.experiences = []
                self.global_solution_history = []
                self.fitness_trial_history = []
                self.population = None

        # self.custom_log()
        return True

    def save_experiences(self):
        action = self.locals.get("actions")[0]
        reward = self.locals.get("rewards")[0]
        done = self.locals.get("dones")[0]
        info = self.locals.get("infos")[0]
        vrptw = self.training_env.envs[0].env.vrp
        global_solution_history = vrptw.global_solution_history
        fitness_trial_history = vrptw.fitness_trial_history
        population = vrptw.population
        if len(global_solution_history) > 0 and len(fitness_trial_history) > 0:
            self.global_solution_history = global_solution_history.copy()
            self.fitness_trial_history = fitness_trial_history.copy()
            self.population = population.copy()
            self.experiences.append(
                dict(action=action, reward=reward, done=done, **info)
            )
        pass

    def save_model(self, mode: str):
        infos = self.locals["infos"][0]
        if "episode" not in infos.keys() or "best_solution" not in infos.keys():
            print("No episode or best_solution info found.")
            return
        global_value = infos["best_solution"]
        episode_reward = infos["episode"]["r"]
        episode_length = infos["episode"]["l"]

        if mode == "solution":
            prefix = "sol"
        elif mode == "reward":
            prefix = "rew"
        else:
            prefix = "int"
        self.model.save(f"{self.save_dir}/{prefix}_{self.num_timesteps:05d}_model")

        # Store experiences
        df_experiences = pd.DataFrame(self.experiences)
        df_experiences.to_excel(
            f"{self.save_dir}/{prefix}_{self.num_timesteps:05d}_exp.xlsx", index=False
        )
        df_experiences.to_pickle(
            f"{self.save_dir}/{prefix}_{self.num_timesteps:05d}_exp.pkl"
        )

        # Save VRPTW state
        with open(
            f"{self.save_dir}/{prefix}_{self.num_timesteps:05d}_vrp.pkl", "wb"
        ) as file:
            pickle.dump(
                dict(
                    global_solution_history=self.global_solution_history,
                    fitness_trial_history=self.fitness_trial_history,
                    population=self.population,
                    episode_reward=episode_reward,
                    episode_length=episode_length,
                    best_solution=global_value,
                    vrptw=self.training_env.envs[0].env.vrp,
                ),
                file,
            )

    def print_info(self, best_solution, global_value, episode_reward, episode_length):
        print(
            f"Step: {self.num_timesteps}, Episode Length: {episode_length}, Episode Reward: {episode_reward:5.3f}, Best Solution: {best_solution:5.3f}, Global Value: {global_value:5.3f}"
        )
        print("----------------------------------------")

    def custom_log(self):
        self.logger.record(
            "train/learning_rate", self.model.lr_schedule(self.progress_remaining)
        )
        current_obs = self.locals["new_obs"]
        current_reward = self.locals["rewards"]
        is_done = self.locals["dones"]
        info_dict = self.locals["infos"]

        self.logger.record("env/current_observation", current_obs)
        self.logger.record("env/current_reward", current_reward)
        self.logger.record("env/is_done", is_done)
        self.logger.record("env/info_dict", str(info_dict))

        # TODO: Not working yet
        if "rollout/ep_rew_mean" in self.logger.name_to_value:
            mean_reward = self.logger.name_to_value["rollout/ep_rew_mean"]
            self.logger.record("env/mean_reward", mean_reward)
        # self.logger.dump(self.num_timesteps)

    def _on_training_end(self) -> None:
        print("Simulation/Training has finished. Executing final callback code.")
        self.save_model(mode="interval")


@dataclass
class SIM_INPUT_PARAMS:
    current_dir: str
    save_interval_seconds: int
    learn_timesteps: int
    run_name: str = ""
    run_description: str = ""
    run_type: str = "NEW"  # Options: "NEW", "LOAD"
    load_folder: str = ""
    file_prefix: str = ""
    load_it: int = 0
    date_prefix: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_dir: str = field(init=False)
    logger_filepath: str = field(init=False)
    monitor_filepath: str = field(init=False)

    def __post_init__(self):
        self.log_dir = f"{self.current_dir}/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger_filepath = f"{self.log_dir}/{self.date_prefix}_training.csv"
        self.monitor_filepath = f"{self.log_dir}/{self.date_prefix}_monitor.csv"

        if self.run_type == "LOAD" and (
            self.load_folder == "" or self.file_prefix == "" or self.load_it == 0
        ):
            raise ValueError(
                "For LOAD run_type, load_folder, load_best_type, and load_it must be specified."
            )


if __name__ == "__main__":
    vpr_input_params = VRPTW_INPUT_PARAMS(
        problem_set="LARGE",  # Options: "SMALL", "LARGE"
        population_size=40,
    )

    sim_input_params = SIM_INPUT_PARAMS(
        run_type="NEW",
        current_dir=pathlib.Path(__file__).parent.resolve(),
        save_interval_seconds=1 * 60,  # 1 minute
        learn_timesteps=20000,
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

    env = RLMH_ENV(vrp=vrptw)
    env = Monitor(
        env,
        filename=sim_input_params.monitor_filepath,
    )

    # This will catch many common issues
    try:
        check_env(env.unwrapped)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    # Initialize the model
    model = SAC("MlpPolicy", env, verbose=1)

    # Set up custom logger
    logger = Logger(
        folder=sim_input_params.log_dir,
        output_formats=[CSVOutputFormat(sim_input_params.logger_filepath)],
    )
    model.set_logger(logger)

    # Define and add the custom callback
    custom_callback = CustomCallback(
        check_freq=1,
        save_dir=f"{sim_input_params.current_dir}/saved_models",
        date_prefix=sim_input_params.date_prefix,
        save_interval_seconds=sim_input_params.save_interval_seconds,
    )

    # Start training
    model.learn(
        total_timesteps=sim_input_params.learn_timesteps, callback=custom_callback
    )

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
