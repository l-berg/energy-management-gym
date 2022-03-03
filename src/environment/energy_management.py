import dataclasses

import gym
from gym import spaces

from src.environment.generation_models import InstLogPlant
import src.environment.variable_power_plants as vp
from src.data_preperation.data_access import EnergyData, WeatherData

from typing import Optional, List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz as tz
import matplotlib.pyplot as plt

PLANT_MODE = 'group'


@dataclass
class EnergyManagementState:
    """Encapsulates the environment's state information at a specific step."""
    step_no: int
    consumption: dataclasses.InitVar[pd.Series]
    residual_generation: np.float32 = 0
    timestamp: datetime = None
    total_load: int = 0
    residual_load: int = 0
    hydro_pump_load: int = 0
    obs: np.array = None       # only for display and debugging purposes
    reward: np.float32 = None  # only for display and debugging purposes
    action: int = None         # only for display and debugging purposes

    def __post_init__(self, consumption):
        """Extract state information from table snapshot."""
        self.total_load = consumption['Total (grid load)[MW]']
        self.residual_load = consumption['Residual load[MW]']
        self.hydro_pump_load = consumption['Hydro pumped storage[MW]']


class EnergyManagementEnv(gym.Env):
    """Custom Environment that follows gym interface.
    Note, that all episodes start at 00:00 CET, but time features use CEST when appropriate. (12 AM becomes 1 AM)

    ### Observation space
    00 | current energy residual load
    01 | current energy production
    02 | current energy production lignite
    03 | current energy production hard coal
    04 | current energy production gas
    05 | current energy production biomass
    06 | current energy production nuclear
    07 | sin(time/year)
    08 | cos(time/year)
    09 | sin(time/day)
    10 | cos(time/day)
    11 | is_weekend
    12 | mean(global radiation)
    13 | mean(wind speed)
    14 | mean(air temperature)
    15 | std(global radiation)
    16 | std(wind speed)
    17 | std(air temperature)
    18 | mean(global radiation in 30 minutes)
    19 | mean(wind speed in 30 minutes)
    20 | mean(air temperature in 30 minutes)
    21 | std(global radiation in 30 minutes)
    22 | std(wind speed in 30 minutes)
    23 | std(air temperature in 30 minutes)
    24 | action affects lignite
    25 | action affects hard coal
    26 | action affects gas
    27 | action affects biomass
    28 | action affects nuclear

    ### Action space
    0 | decrease plant output by 5% of total capacity
    1 | decrease plant output by 2.5% of total capacity
    2 | do nothing
    3 | increase plant output by 2.5% of total capacity
    4 | increase plant output by 5% of total capacity
    """
    metadata = {'render.modes': ['human']}
    state_history = []
    obs_timezone = tz.timezone('Europe/Berlin')
    weather_labels = ['radiation_global', 'wind_speed', 'temperature_air_mean_200']

    def __init__(self, episode_length: timedelta = None, step_period: timedelta = None,
                 start_date: Union[str, datetime] = None, end_date: Union[str, datetime] = None,
                 use_residual_load=True, use_weather_data=True, use_time_data=True,
                 relative_control=True, num_actions=5, action_range=0.05, solar_output_scale=1,
                 wind_output_scale=1, output_diff_scale=3):
        """
        Given dates are interpreted as of timezone 'Europe/Berlin'.

        :param episode_length: length of an episode (default: 1 day)
        :param step_period: step size (default: 15 minutes)
        :param start_date: specifies start of interval to use real world data from (inclusive, default: 2015-01)
        :param end_date: end of interval of real world data (exclusive, default: 2022-01)
        """
        super(EnergyManagementEnv, self).__init__()

        # episode length
        if episode_length is None:
            self.episode_length = timedelta(days=1)
        else:
            self.episode_length = episode_length

        # step period
        if step_period is None:
            self.step_period = timedelta(minutes=15)
        else:
            self.step_period = step_period

        # start date
        default_start_date = pd.Timestamp('2015-01').tz_localize(self.obs_timezone).tz_convert('UTC').to_pydatetime()
        if start_date is None:
            start_date = '2015-01'
        self.start_date = pd.Timestamp(start_date).tz_localize(self.obs_timezone).tz_convert('UTC').to_pydatetime()
        if self.start_date < default_start_date:
            raise IndexError(f'Given start date {self.start_date} comes before 2015-01')

        # end date
        default_end_date = pd.Timestamp('2022-01').tz_localize(self.obs_timezone).tz_convert('UTC').to_pydatetime()
        if end_date is None:
            end_date = '2022-01'
        self.end_date = pd.Timestamp(end_date).tz_localize(self.obs_timezone).tz_convert('UTC').to_pydatetime()
        if self.end_date > default_end_date:
            raise IndexError(f'Given end date {self.end_date} comes after 2022-01')

        # draw conclusions for #steps and #episodes
        self.steps_per_episode = int(self.episode_length / self.step_period)
        if self.steps_per_episode * self.step_period != self.episode_length:
            raise ValueError(f'Step size {self.step_period} does not evenly divide episode length {self.episode_length}')
        self.episodes_in_interval = int((self.end_date - self.start_date) / self.episode_length)
        if self.episodes_in_interval * self.episode_length != self.end_date - self.start_date:
            print(f'Warning: Episode length {self.episode_length} does not evenly divide chosen interval from {self.start_date} to {self.end_date}')

        self.action_space = spaces.Discrete(num_actions)

        float_max = np.finfo(np.float32).max
        high = np.array([
            float_max, float_max,
            float_max, float_max, float_max, float_max, float_max,
            1, 1,
            1, 1,
            True,
            *[float_max] * len(self.weather_labels) * 2 * 2,
            1, 1, 1, 1, 1,
        ], dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.random_generator = np.random.default_rng()

        # import real-world electrical energy dataDE
        self.generation_data = EnergyData('datasets/generation/generation.csv')
        self.consumption_data = EnergyData('datasets/consumption/consumption.csv')
        self.capacity_data = EnergyData('datasets/installed_generation/installed_generation.csv')
        self.weather_mean = WeatherData('datasets/weather_fine/weather_mean.csv')
        self.weather_std = WeatherData('datasets/weather_fine/weather_std.csv')

        self.current_time = None
        self.step_counter = 0
        self.sub_step = 0
        self.state = None
        self.plants = []

        # init episode stats
        self.total_production = 0
        self.total_co2_emissions = 0
        self.total_overproduction = 0
        self.total_underproduction = 0

        # settings for experiments
        self.use_residual_load = use_residual_load
        self.use_weather_data = use_weather_data
        self.use_time_data = use_time_data
        self.relative_control = relative_control
        self.num_actions = num_actions
        self.action_range = action_range
        self.solar_output_scale = solar_output_scale
        self.wind_output_scale = wind_output_scale
        self.output_diff_scale = output_diff_scale

    def _residual_load_scale(self):
        """As wind and solar output increase, the residual load shrinks."""
        consumption = self.consumption_data[self.current_time]
        generation = self.generation_data[self.current_time]

        original_residual_load = consumption['Residual load[MW]']
        total_load = consumption['Total (grid load)[MW]']

        wind_gen = (generation['Wind offshore[MW]'] + generation['Wind onshore[MW]']) * self.wind_output_scale
        solar_gen = generation['Photovoltaics[MW]'] * self.solar_output_scale

        new_residual_load = max(total_load - (wind_gen + solar_gen), 0)
        return new_residual_load / original_residual_load

    def _update_state(self, action):
        """Advance the environment's state by one step."""
        # signal current plant to change output
        if self.relative_control:
            output_delta = action * 2 * self.action_range / (self.num_actions - 1) - self.action_range
            self.plants[self.sub_step].change_output(output_delta)
        else:
            self.plants[self.sub_step].set_output(action / (self.num_actions - 1))

        self.sub_step += 1
        # only update state after all plants have been updated
        if self.sub_step == len(self.plants):
            self.current_time += self.step_period
            self.step_counter += 1
            self.sub_step = 0
            consumption_snapshot = self.consumption_data[self.current_time]
            consumption_snapshot['Residual load[MW]'] *= self._residual_load_scale()
            residual_generation = 0
            for plant in self.plants:
                residual_generation += plant.step()
            self.state = EnergyManagementState(step_no=self.step_counter, timestamp=self.current_time,
                                               consumption=consumption_snapshot, residual_generation=residual_generation)
            # increment totals
            part_of_hour = self.step_period.total_seconds() / 3600
            for plant in self.plants:
                self.total_co2_emissions += plant.get_costs() * part_of_hour

            self.total_production += self.state.residual_generation * part_of_hour
            diff = self.state.residual_generation - self.state.residual_load
            if diff > 0:
                self.total_overproduction += abs(diff) * part_of_hour
            else:
                self.total_underproduction += abs(diff) * part_of_hour

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        self._update_state(action)
        # state has been updated, now compute return values
        observation = self._obs()

        # calculate reward
        max_cost = max(list(p.price_per_mwh for p in self.plants))
        diff = (self.state.residual_generation - self.state.residual_load) / 10000.0
        reward = max_cost * self.state.residual_load / 10000.0
        for plant in self.plants:
            reward -= plant.get_costs() / 10000.0
        if diff < 0:
            reward += self.output_diff_scale * diff * max_cost
        else:
            reward -= diff * max_cost

        done = (self.step_counter >= self.steps_per_episode)
        info = {}

        # for visualization and debugging
        self.state.action = action
        self.state.obs = observation
        self.state.reward = reward

        return observation, reward, done, info

    def _time_obs(self):
        """Build time features for observation."""
        d = self.state.timestamp.astimezone(self.obs_timezone).replace(tzinfo=None)

        # extract time-of-year information
        year_date = datetime(d.year, 1, 1)
        one_year = datetime(d.year+1, 1, 1) - year_date
        portion_of_year = (d - year_date) / one_year

        # weekend information
        is_weekend = (d.weekday() >= 5)  # saturday or sunday (mon==0, ..., sun==6)

        # extract time-of-day information
        time_of_day = timedelta(hours=d.hour, minutes=d.minute, seconds=d.second)
        one_day = timedelta(days=1)
        portion_of_day = time_of_day / one_day

        w0 = 2*np.pi
        obs = np.array([
            np.sin(w0 * portion_of_year),
            np.cos(w0 * portion_of_year),
            np.sin(w0 * portion_of_day),
            np.cos(w0 * portion_of_day),
            is_weekend
        ], dtype=np.float32)
        if not self.use_time_data:
            return np.zeros_like(obs)
        return obs

    def _energy_obs(self):
        """Build residuals and generation into observation."""

        # normalize energy totals
        eps = np.finfo(np.float32).eps.item()
        residual_load_norm = (self.state.residual_load - self.consumption_data.mean['Residual load[MW]']
                              ) / (self.consumption_data.std['Residual load[MW]'] + eps)
        residual_gen_norm = (self.state.residual_generation - self.consumption_data.mean['Residual load[MW]']
                             ) / (self.consumption_data.std['Residual load[MW]'] + eps)

        # get current output for all plants
        current_outputs = np.zeros(len(self.plants))
        for i in range(len(current_outputs)):
            current_outputs[i] = self.plants[i].current_output / self.plants[i].max_capacity

        if not self.use_residual_load:
            residual_load_norm = np.zeros_like(residual_load_norm)
        obs = np.array([residual_load_norm, residual_gen_norm], dtype=np.float32)
        obs = np.append(obs, current_outputs)
        return obs

    def _weather_obs(self, time_offset=None):
        """Build weather data into observation."""
        if not self.use_weather_data:
            return np.zeros(2 * len(self.weather_labels))

        if time_offset is not None:
            d = self.current_time + time_offset
        else:
            d = self.current_time

        eps = np.finfo(np.float32).eps.item()
        current_mean = self.weather_mean[d]
        current_std = self.weather_std[d]

        weather_features = []
        for label in self.weather_labels:
            weather_features.append(
                (current_mean[label] - self.weather_mean.mean[label]) / (self.weather_mean.std[label] + eps)
            )
        for label in self.weather_labels:
            weather_features.append(
                (current_std[label] - self.weather_std.mean[label]) / (self.weather_std.std[label] + eps)
            )
        return np.array(weather_features, dtype=np.float32)

    def _obs(self):
        """Extract an observation (=features) from the current state."""

        # collect observations
        energy_obs = self._energy_obs()
        time_obs = self._time_obs()
        weather_now_obs = self._weather_obs()
        weather_forecast_obs = self._weather_obs(timedelta(minutes=15))

        # one hot encode active plant
        active_plant_obs = np.zeros(len(self.plants), dtype=np.float32)
        active_plant_obs[self.sub_step - 1] = 1

        # construct feature vector
        obs = np.concatenate([energy_obs, time_obs, weather_now_obs, weather_forecast_obs, active_plant_obs])

        # update state only if all plants have been updated
        if self.sub_step == 0:
            self.state_history.append(self.state)
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.step_counter = 0
        self.sub_step = 0
        self.state_history = []

        self.total_production = 0
        self.total_co2_emissions = 0
        self.total_overproduction = 0
        self.total_underproduction = 0

        # random starting time
        episode_no = self.random_generator.integers(self.episodes_in_interval)
        self.current_time = self.start_date + episode_no * self.episode_length

        # initialize plants with current year capacity and current date output
        generation = self.generation_data[self.current_time]
        capacity = self.capacity_data[self.current_time]

        # residual generation is expected to shrink with residual load
        generation *= self._residual_load_scale()

        step_minutes = self.step_period.total_seconds() / 60
        self.plants = [
            vp.LignitePowerPlant(generation['Lignite[MW]'], capacity['Lignite[MW]'], step_minutes, PLANT_MODE),
            vp.HardCoalPowerPlant(generation['Hard coal[MW]'], capacity['Hard coal[MW]'], step_minutes, PLANT_MODE),
            vp.GasPowerPlant(generation['Fossil gas[MW]'], capacity['Fossil gas[MW]'], step_minutes, PLANT_MODE),
            vp.BioMassPowerPlant(generation['Biomass[MW]'], capacity['Biomass[MW]'], step_minutes, PLANT_MODE),
            vp.NuclearPowerPlant(generation['Nuclear[MW]'], capacity['Nuclear[MW]'], step_minutes, PLANT_MODE)
        ]

        # set initial generation
        initial_generation = 0
        for plant in self.plants:
            initial_generation += plant.current_output

        # finally, summarize everything in new state
        consumption_snapshot = self.consumption_data[self.current_time]
        consumption_snapshot['Residual load[MW]'] *= self._residual_load_scale()
        self.state = EnergyManagementState(step_no=self.step_counter, timestamp=self.current_time,
                                           consumption=consumption_snapshot, residual_generation=initial_generation)

        return self._obs()  # reward, done, info can't be included

    def render(self, mode='human'):
        fig, ax= plt.subplots(nrows=3, ncols=2, sharex=True, dpi=100, figsize=(16,8))  # (width, height)
        energy_ax = ax[0, 0]
        reward_ax = ax[1, 0]
        episode_start = self.state_history[0].timestamp.astimezone(self.obs_timezone).replace(tzinfo=None)
        fig.suptitle(f'Energy Management starting {str(episode_start)}')
        # all_steps = list(range(len(self.state_history)))

        # plot generated energy and load
        props_of_interest = ['residual_generation', 'total_load', 'residual_load']
        energy_values = {}
        for p in props_of_interest:
            steps, values = [], []
            for state in self.state_history:
                value = getattr(state, p)
                if value is not None:
                    steps.append(state.step_no)
                    values.append(value / 1000)
            energy_values[p] = (steps, values)

        for label, (steps, values) in energy_values.items():
            energy_ax.plot(steps, values, label=label)
        energy_ax.legend()
        energy_ax.set_ylabel('power [GW]')
        energy_ax.set_title('Electrical Energy')

        # plot observations
        time_labels = [
            'sin(time / year)',
            'cos(time / year)',
            'sin(time / day)',
            'cos(time / day)',
            'is_weekend',
        ]
        energy_labels = [
            'residual load',
            'residual gen',
            'lignite gen',
            'hard coal gen',
            'gas gen',
            'biomass gen',
            'nuclear gen'
        ]
        feature_steps = []
        feature_list = []
        for state in self.state_history:
            if state.obs is not None:
                feature_steps.append(state.step_no)
                feature_list.append(state.obs)
        feature_mat = np.stack(feature_list, axis=1)

        def plot_mat(feature_ax, labels, mat, name):
            for label, feature_values in zip(labels, mat):
                feature_ax.plot(feature_steps, feature_values, label=label)
            feature_ax.legend()
            feature_ax.set_title(f'{name} Observation')
        plot_mat(ax[0, 1], energy_labels, feature_mat, 'Energy')
        plot_mat(ax[1, 1], time_labels, feature_mat[len(energy_labels):], 'Time')

        # weather observations (only now, not forecast)
        for i, l in enumerate(self.weather_labels):
            mean_values = feature_mat[len(energy_labels)+len(time_labels)+i]
            std_values = feature_mat[len(energy_labels)+len(time_labels)+i+len(self.weather_labels)]
            weather_ax = ax[2, 1]
            weather_ax.errorbar(feature_steps, mean_values, yerr=std_values, label=l)
            weather_ax.legend()
            weather_ax.set_title('Weather Observation')

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

        total_reward = 0
        for state in self.state_history:
            total_reward += state.reward
        stats_ax = ax[2, 0]
        stats_ax.set_axis_off()
        stats_text = ''
        stats_text += f'Total production:    {self.total_production/1000 :6.1f} GWh\n'
        stats_text += f'    Overproduction:  {self.total_overproduction/1000 :6.1f} GWh\n'
        stats_text += f'    Underproduction: {self.total_underproduction/1000 :6.1f} GWh\n'
        stats_text += f'Total CO2 emissions: {self.total_co2_emissions/1000 :6.1f} kt\n\n'
        stats_text += f'Timespan: {str(self.step_period * self.step_counter)}\n\n'
        stats_text += f'Total Reward: {total_reward:.1f}'
        stats_ax.text(0, 0.5, stats_text, ha='left', ma='left', va='center',
                      family='monospace', size=14, transform=stats_ax.transAxes)

        fig.tight_layout()
        plt.show()

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        self.random_generator = np.random.default_rng(seed)
        return [seed]
