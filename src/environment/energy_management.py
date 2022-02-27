import dataclasses

import gym
from gym import spaces

from src.environment.generation_models import InstLogPlant
import src.environment.variable_power_plants as vp

from typing import Optional, List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import datetime
import locale
import matplotlib.pyplot as plt

STEP_SIZE = 15
SUB_STEPS = 5
STEPS_PER_EPISODE = 24 * 60 / STEP_SIZE
NUM_ACTIONS = 5
ACTION_RANGE = 0.1


def time_of_row(row):
    time = datetime.datetime.strptime(f"{row['Date']} {row['Time of day']}", "%b %d %Y %I:%M %p")
    return time


@dataclass
class EnergyManagementState:
    """Encapsulates the environment's state information at a specific step."""
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
        """Extract state information from table row."""
        self.timestamp = time_of_row(consumption_row)
        self.total_load = consumption_row['Total (grid load)[MWh]']
        self.residual_load = consumption_row['Residual load[MWh]']
        self.hydro_pump_load = consumption_row['Hydro pumped storage[MWh]']


class EnergyManagementEnv(gym.Env):
    """Custom Environment that follows gym interface.
    Note, that all episodes start at 00:00 CET, but time features use CEST when appropriate. (12 AM becomes 1 AM)

    ### Observation space
    00 | sin(time/year)
    01 | cos(time/year)
    02 | sin(time/day)
    03 | cos(time/day)
    04 | is_weekend
    05 | current energy residual load
    06 | current energy production
    TODO
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

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # np.finfo(np.float32).max,
        float_max = np.finfo(np.float32).max
        high = np.array([
            1, 1,
            1, 1,
            True,
            float_max, float_max,
            float_max, float_max, float_max, float_max, float_max,
            1, 1, 1, 1, 1,
        ], dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.random_generator = np.random.default_rng()

        # import real-world electrical energy dataDE
        generation_path = 'datasets/generation/generation.csv'
        self.generation_table = pd.read_csv(generation_path, sep=';')
        self.generation_description = pd.read_csv(f'{generation_path[:-4]}_description.csv', sep=';', index_col=0)
        consumption_path = 'datasets/consumption/consumption.csv'
        self.consumption_table = pd.read_csv(consumption_path, sep=';')
        self.consumption_description = pd.read_csv(f'{consumption_path[:-4]}_description.csv', sep=';', index_col=0)
        capacity_path = 'datasets/installed_generation/installed_generation_all.csv'
        self.capacity_table = pd.read_csv(capacity_path, sep=";")

        self.step_counter = 0
        self.sub_step = 0
        self.state = None
        self.plants = []

    def _update_state(self, action):
        """Advance the environment's state by one step."""
        # signal current plant to change output
        mid_point = int(NUM_ACTIONS / 2)
        if action < mid_point:
            self.plants[self.sub_step].less((ACTION_RANGE / mid_point) * (action + 1))
        elif action > mid_point:
            self.plants[self.sub_step].more((ACTION_RANGE / mid_point) * (action - mid_point))
        elif NUM_ACTIONS % 2 == 0:
            self.plants[self.sub_step].more((ACTION_RANGE / mid_point))

        self.sub_step += 1

        # only update state after all plants have been updated
        if self.sub_step == SUB_STEPS:
            self.step_counter += 1
            self.sub_step = 0
            next_table_index = self.state.table_index + 1
            consumption_row = self.consumption_table.loc[next_table_index]
            residual_generation = 0
            for plant in self.plants:
                residual_generation += plant.step()
            self.state = EnergyManagementState(step_no=self.step_counter, table_index=next_table_index,
                                               consumption_row=consumption_row, residual_generation=residual_generation)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        self._update_state(action)
        # state has been updated, now compute return values
        observation = self._obs()

        # calculate reward
        max_cost = max(list(p.price_per_mwh for p in self.plants))
        diff = (self.state.residual_generation - self.state.residual_load) / 1000  # [diff] = GWh
        reward = max_cost * self.state.residual_load / 1000.0
        for plant in self.plants:
            reward -= plant.get_costs() / 1000.0
        if diff < 0:
            reward += 3 * diff * max_cost
        else:
            reward -= diff

        done = (self.step_counter >= STEPS_PER_EPISODE)
        info = {}

        # for visualization and debugging
        self.state.action = action
        self.state.obs = observation
        self.state.reward = reward

        return observation, reward, done, info

    def _obs(self):
        """Extract an observation (=features) from the current state."""
        d = self.state.timestamp

        # extract time-of-year information
        year_date = datetime.datetime(d.year, 1, 1, 0, 0)
        one_year = datetime.datetime(d.year+1, 1, 1, 0, 0) - year_date
        sec_of_year = (d - year_date).total_seconds()
        portion_of_year = sec_of_year / one_year.total_seconds()

        # weekend information
        # portion_of_week = d.weekday() / 7
        is_weekend = (d.isoweekday() >= 6)  # saturday or sunday (ISO: mon==1, ..., sun==7)

        # extract time-of-day information
        sec_of_day = datetime.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second).total_seconds()
        one_day = datetime.timedelta(days=1)
        portion_of_day = sec_of_day / one_day.total_seconds()

        # normalize energy totals
        eps = np.finfo(np.float32).eps.item()
        residual_load_norm = (self.state.residual_load - self.consumption_description['Residual load[MWh]'].loc['mean']
                              ) / (self.consumption_description['Residual load[MWh]'].loc['std'] + eps)
        residual_gen_norm = (self.state.residual_generation - self.consumption_description['Residual load[MWh]'].loc['mean']
                             ) / (self.consumption_description['Residual load[MWh]'].loc['std'] + eps)

        # one hot encode active plant
        active_plant = np.zeros(len(self.plants))
        active_plant[self.sub_step - 1] = 1

        # get current output for all plants
        generation = np.zeros(len(self.plants))
        for i in range(len(generation)):
            if self.state.residual_generation == 0:
                generation[i] = 0
            else:
                generation[i] = self.plants[i].target_output / self.state.residual_generation

        # construct feature vector
        w0 = 2*np.pi
        obs = np.array([
            np.sin(w0 * portion_of_year),
            np.cos(w0 * portion_of_year),
            np.sin(w0 * portion_of_day),
            np.cos(w0 * portion_of_day),
            is_weekend,
            residual_load_norm,
            residual_gen_norm,
        ], dtype=np.float32)
        obs = np.append(obs, generation)
        obs = np.append(obs, active_plant)

        if self.sub_step == 0:
            self.state_history.append(self.state)  # maybe also include obs and reward for visualization?
        return obs

    def reset(self, seed = None):
        if seed is not None:
            self.seed(seed)

        self.step_counter = 0
        self.sub_step = 0
        self.state_history = []

        table_index = self.random_generator.integers(self.consumption_table.shape[0]/STEPS_PER_EPISODE) * STEPS_PER_EPISODE
        consumption_row = self.consumption_table.loc[table_index]
        self.state = EnergyManagementState(step_no=self.step_counter, table_index=table_index, consumption_row=consumption_row)

        # initialize plants with current year capacity and current date output
        generation_row = self.generation_table.loc[table_index]
        current_year = self.state.timestamp.year
        for r in range(self.capacity_table.shape[0]):
            if time_of_row(self.capacity_table.loc[r]).year == current_year:
                capacity_row = self.capacity_table.loc[r]

        step_time = 60 / STEP_SIZE
        self.plants = [vp.LignitePowerPlant(generation_row['Lignite[MWh]'], capacity_row['Lignite[MW]'] / step_time, STEP_SIZE),
                       vp.HardCoalPowerPlant(generation_row['Hard coal[MWh]'], capacity_row['Hard coal[MW]'] / step_time, STEP_SIZE),
                       vp.GasPowerPlant(generation_row['Fossil gas[MWh]'], capacity_row['Fossil gas[MW]'] / step_time, STEP_SIZE),
                       vp.BioMassPowerPlant(generation_row['Biomass[MWh]'], capacity_row['Biomass[MW]'] / step_time, STEP_SIZE),
                       vp.NuclearPowerPlant(generation_row['Nuclear[MWh]'], capacity_row['Nuclear[MW]'] / step_time, STEP_SIZE)]

        return self._obs()  # reward, done, info can't be included
    
    def render(self, mode='human'):
        fig, (energy_ax, feature_ax, reward_ax) = plt.subplots(3, dpi=100, figsize=(8,8))  # (width, height)
        fig.suptitle(f'Energy Management on {str(self.state_history[0].timestamp.date())}')
        steps = list(range(len(self.state_history)))

        # plot generated energy and load
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

        # plot observations
        feature_labels = [
            'sin(time / year)',
            'cos(time / year)',
            'sin(time / day)',
            'cos(time / day)',
            'is_weekend',
            'residual load',
            'residual gen',
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

        # plot actions and rewards in one
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
        action_ax.scatter(action_steps, action_values, label='action', color=action_color, marker='x')
        action_ax.tick_params(axis='y', labelcolor=action_color)

        fig.tight_layout()
        plt.show()

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        self.random_generator = np.random.default_rng(seed)
        return [seed]
