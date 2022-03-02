import pytest
import src.environment.energy_management as em
from src.utils.utils import set_seed

from stable_baselines3.common.env_checker import check_env

import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def run_episode(env):
    env.reset()
    done = False
    step_no = 0
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        step_no += 1
    return step_no


def test_episode_length():
    expected_steps = 4*24*5

    env = em.EnergyManagementEnv()
    env.reset()

    step_counter = 0
    for _ in range(expected_steps+5):
        _, _, done, _ = env.step(0)
        step_counter += 1
        if done:
            break

    assert step_counter == expected_steps


def test_reproducibility():
    set_seed(99)

    # random values without env
    r_torch = torch.rand((1,)).item()
    r_numpy = np.random.rand()
    r_random = random.random()

    set_seed(99)
    env = em.EnergyManagementEnv()
    env.reset()

    # run the environment a bit
    for _ in range(10):
        env.step(0)
    env.seed(20)
    env.reset()
    for _ in range(10):
        env.step(0)
    obs_11, _, _, _ = env.step(0)

    # running the environment should not have affected the randomness
    assert r_torch == torch.rand((1,)).item()
    assert r_numpy == np.random.rand()
    assert r_random == random.random()

    # running the environment with the same seed should lead to the same observation
    env = em.EnergyManagementEnv()
    env.seed(20)
    env.reset()
    for _ in range(10):
        env.step(0)
    assert (obs_11 == env.step(0)[0]).all()


def test_stable_baselines_compatibility():
    env = em.EnergyManagementEnv()

    # stable baselines3 test suite (checks e.g. action and observation space)
    check_env(env)


def test_data_coherency():
    def verify_description(basepath):
        descpath = f'{basepath[:-4]}_description.csv'

        df = pd.read_csv(basepath, sep=';')
        desc = pd.read_csv(descpath, sep=';', index_col=0)
        assert np.isclose(df.describe(), desc).all()

    verify_description('datasets/generation/generation.csv')
    verify_description('datasets/consumption/consumption.csv')


def test_time_edges():
    # test start of dataset
    env = em.EnergyManagementEnv(start_date='2015-01-01', end_date='2015-01-01T01:00:00',
                                 episode_length=timedelta(hours=1), step_period=timedelta(minutes=5))
    assert env.steps_per_episode == 60 / 5
    assert env.episodes_in_interval == 1

    assert run_episode(env) == 60 / 5 * 5  # last *5 for substeps

    # test end of dataset
    env = em.EnergyManagementEnv(start_date='2021-12-31T12', end_date='2022-01-01',
                                 episode_length=timedelta(hours=12), step_period=timedelta(minutes=5))
    assert env.episodes_in_interval == 1
    assert run_episode(env) == 12 * 60 / 5 * 5  # last *5 for substeps


def test_data_samples():
    # manual lookup: Feb 24, 2017, 2:30 AM offshore wind production was 808 MWh in 15 min -> 4*808 MW of power
    env = em.EnergyManagementEnv(start_date='2017-02-24T02:30', end_date='2017-02-25T02:30')
    assert env.episodes_in_interval == 1

    env.reset()
    assert env.generation_data[env.current_time]['Wind offshore[MW]'] == 808 * 4

    # manual lookup: May 5, 2021, 10:00 AM residual load was 5,210 MWh per 15 min
    # manual lookup: May 5, 2021, 10:15 AM residual load was 4,940 MWh per 15 min
    # -> at 10:05 AM, linearly interpolated load should be ((2/3)*5210 + (1/3)*4940) * 4
    env = em.EnergyManagementEnv(start_date='2021-05-05T10:05', end_date='2021-05-06T10:05')
    assert env.episodes_in_interval == 1

    env.reset()
    assert env.consumption_data[env.current_time]['Residual load[MW]'] == pytest.approx(((2/3)*5210 + (1/3)*4940) * 4)



