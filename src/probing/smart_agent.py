# https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/

import src.environment.energy_management as em

import gym
from stable_baselines3 import PPO
from pathlib import Path
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'show'], help='Train from scratch or visualize existing checkpoint')
parser.add_argument('steps', type=int)
args = parser.parse_args()

models_dir = "checkpoints/simple_ppo"
logdir = "tensorboard_logs"

def train():
    # create dirs if they do not exist
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)

    env = em.EnergyManagementEnv()
    env.reset(seed=0)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000
    max_iterations = math.ceil(args.steps / TIMESTEPS)
    for i in range(1, max_iterations+1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

def visualize():
    env = em.EnergyManagementEnv()
    env.reset()  # do not seed so we can call it multiple times

    model_path = f"{models_dir}/{args.steps}.zip"
    model = PPO.load(model_path, env=env)

    episodes = 5

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        env.render()

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        visualize()
