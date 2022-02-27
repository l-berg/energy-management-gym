import src.environment.energy_management as em
from src.utils.utils import set_seed

from stable_baselines3.common.env_checker import check_env

import torch
import random
import numpy as np
import pandas as pd


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

    # running the environment should not have affected the randomness
    assert r_torch == torch.rand((1,)).item()
    assert r_numpy == np.random.rand()
    assert r_random == random.random()


def test_stable_baselines_compatibility():
    env = em.EnergyManagementEnv()
    check_env(env)


def test_data_coherency():
    def verify_description(basepath):
        descpath = f'{basepath[:-4]}_description.csv'

        df = pd.read_csv(basepath, sep=';')
        desc = pd.read_csv(descpath, sep=';', index_col=0)
        assert np.isclose(df.describe(), desc).all()

    verify_description('datasets/generation/generation.csv')
    verify_description('datasets/consumption/consumption.csv')


