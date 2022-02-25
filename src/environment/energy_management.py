import dataclasses

import gym
from gym import spaces

from src.environment.generation_models import InstLogPlant

from typing import Optional, List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import datetime
import locale
import matplotlib.pyplot as plt

STEPS_PER_EPISODE = 24 * 4


def time_of_row(row):
    time = datetime.datetime.strptime(f"{row['Date']} {row['Time of day']}", "%b %d, %Y %I:%M %p")
    return time


@dataclass
class EnergyManagementState:
    step_no: int
    table_index: int
    consumption_row: dataclasses.InitVar
    residual_generation: np.float32 = 0
    timestamp: datetime.datetime = None
    total_load: int = 0
    residual_load: int = 0
    hydro_pump_load: int = 0
    obs: np.array = None       # only for display and debugging purposes
    reward: np.float32 = None  # only for display and debugging purposes
    action: int = None         # only for display and debugging purposes

    def __post_init__(self, consumption_row):
        self.timestamp = time_of_row(consumption_row)
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        self.total_load = locale.atoi(consumption_row['Total (grid load)[MWh]'])
        self.residual_load = locale.atoi(consumption_row['Residual load[MWh]'])
        self.hydro_pump_load = locale.atoi(consumption_row['Hydro pumped storage[MWh]'])


class EnergyManagementEnv(gym.Env):
    """Custom Environment that follows gym interface

    ### Observation space
    00 | sin(time/year)
    01 | cos(time/year)
    02 | sin(time/day)
    03 | cos(time/day)
    04 | is_weekend
    TODO
    07 | current energy consumption
    08 | current energy production
    09 | current energy production solar
    10 | current energy production wind
    11 | current energy consumption hydro storage
    12 | current energy production hydro storage
    000 | action affects slow fossil
    001 | action affects fast fossil
    002 | action affects hydro storage

    ### Action space
    1 | higher output
    2 | lower output
    """
    metadata = {'render.modes': ['human']}
    state_history = []

    def __init__(self):
        super(EnergyManagementEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        # np.finfo(np.float32).max,
        high = np.array([
            1, 1,
            1, 1,
            True,
            ], dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.random_generator = np.random.default_rng()

        self.generation_table = pd.read_csv('datasets/generation/generation_all.csv', sep=';')
        self.consumption_table = pd.read_csv('datasets/consumption/consumption_all.csv', sep=';')

        self.step_counter = 0
        self.state = None
        self.plant = None

    def _update_state(self, action):
        if action == 0:
            self.plant.less()
        else:
            self.plant.more()

        self.step_counter += 1
        next_table_index = self.state.table_index + 1
        consumption_row = self.consumption_table.loc[next_table_index]

        residual_generation = self.plant.step()
        self.state = EnergyManagementState(step_no=self.step_counter, table_index=next_table_index,
                                           consumption_row=consumption_row, residual_generation=residual_generation)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        self._update_state(action)

        # state has been updated, now compute return values
        observation = self._obs()

        diff = self.state.residual_generation - self.state.residual_load
        if diff > 0:
            reward = -diff
        else:
            reward = 10*diff

        done = (self.step_counter >= STEPS_PER_EPISODE)
        info = {}

        # for visualization and debugging
        self.state.action = action
        self.state.obs = observation
        self.state.reward = reward

        return observation, reward, done, info

    def _obs(self):
        """Extract an observation (=features) from the current state"""
        d = self.state.timestamp

        year_date = datetime.datetime(d.year, 1, 1, 0, 0)
        one_year = datetime.datetime(d.year+1, 1, 1, 0, 0) - year_date
        sec_of_year = (d - year_date).total_seconds()
        portion_of_year = sec_of_year / one_year.total_seconds()

        # portion_of_week = d.weekday() / 7
        is_weekend = (d.isoweekday() >= 6)  # saturday or sunday (ISO: mon==1, ..., sun==7)

        sec_of_day = datetime.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second).total_seconds()
        one_day = datetime.timedelta(days=1)
        portion_of_day = sec_of_day / one_day.total_seconds()

        w0 = 2*np.pi
        obs = np.array([
            np.sin(w0 * portion_of_year),
            np.cos(w0 * portion_of_year),
            np.sin(w0 * portion_of_day),
            np.cos(w0 * portion_of_day),
            is_weekend,
        ], dtype=np.float32)

        self.state_history.append(self.state)  # maybe also include obs and reward for visualization?
        return obs

    def reset(self, seed = None):
        if seed is not None:
            self.seed(seed)

        self.step_counter = 0
        self.state_history = []

        table_index = self.random_generator.integers(self.consumption_table.shape[0]/STEPS_PER_EPISODE) * STEPS_PER_EPISODE
        consumption_row = self.consumption_table.loc[table_index]
        self.state = EnergyManagementState(step_no=self.step_counter, table_index=table_index, consumption_row=consumption_row)

        self.plant = InstLogPlant(self.state.residual_load, 100000)

        return self._obs()  # reward, done, info can't be included
    
    def render(self, mode='human'):
        fig, (energy_ax, feature_ax, reward_ax) = plt.subplots(3, dpi=100, figsize=(8,8))  # (width, height)
        fig.suptitle(f'Energy Management on {str(self.state_history[0].timestamp.date())}')
        steps = list(range(len(self.state_history)))

        props = ['residual_generation', 'total_load', 'residual_load']
        energy_values = {}
        for p in props:
            values = []
            for state in self.state_history:
                values.append(getattr(state, p))
            energy_values[p] = values

        for label, values in energy_values.items():
            energy_ax.plot(steps, values, label=label)
        energy_ax.legend()
        energy_ax.set_title('Electrical Energy')

        feature_labels = [
             'sin(time / year)',
             'cos(time / year)',
             'sin(time / day)',
             'cos(time / day)',
             'is_weekend',
        ]
        feature_steps = []
        feature_list = []
        for state in self.state_history:
            if state.obs is not None:
                feature_steps.append(state.step_no)
                feature_list.append(state.obs)
        feature_mat = np.stack(feature_list, axis=1)

        for label, feature_values in zip(feature_labels, feature_mat):
            feature_ax.plot(feature_steps, feature_values, label=label)
        feature_ax.legend()
        feature_ax.set_title('Observation')

        reward_steps = []
        reward_values = []
        action_steps = []
        action_values = []
        for state in self.state_history:
            if state.reward is not None:
                reward_steps.append(state.step_no)
                reward_values.append(state.reward)
            if state.action is not None:
                action_steps.append(state.step_no)
                action_values.append(state.action)

        reward_ax.plot(reward_steps, reward_values, label='reward')
        reward_ax.set_ylabel('reward')
        reward_ax.set_title('Rewards/Actions')

        action_color = 'green'
        action_ax = reward_ax.twinx()
        action_ax.set_ylabel('action', color=action_color)
        action_ax.plot(action_steps, action_values, label='action', color=action_color)
        action_ax.tick_params(axis='y', labelcolor=action_color)

        fig.tight_layout()
        plt.show()


    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        self.random_generator = np.random.default_rng(seed)
        return seed
