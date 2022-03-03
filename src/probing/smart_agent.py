# https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/ TODO
# Energy for 100k steps: 50312 J ~ 14Wh

import src.environment.energy_management as em

import gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
import argparse
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", choices=["auto", "manual"], default="manual",
                    help="Use auto in combination with a predefined experiment or manually set parameters")
parser.add_argument('-a', '--action', choices=['train', 'show'],
                    help='Train from scratch or visualize existing checkpoint')
parser.add_argument("-e", "--experiment", type=int, choices=[0, 1], default=-1, help="Predefined experiment to run")
parser.add_argument('-s', '--steps', type=int, default=100000)
parser.add_argument("-S", "--seed", type=int, help="Seed used for the environment")
parser.add_argument("-A", "--algorithm", choices=["PPO", "DQL"], default="PPO",
                    help="Select desired RL algorithm")
args = parser.parse_args()
# default parameters
SB3_PARAMS = {
    "policy": "MlpPolicy",
    "verbose": 1
}
# contains all parameters for predefined experiments
EXPERIMENTS = [{
    "algorithm": "PPO",
    "steps": 10000,
    "sb3_params": {
        "policy": "MlpPolicy",
        "verbose": 0
    }
}, {
    "algorithm": "DQN",
    "steps": 300000,
    "sb3_params": {
        "policy": "MlpPolicy",
        "verbose": 1
    }
}]


def train():
    env = em.EnergyManagementEnv()
    mean_rewards = []
    reward_stds = []
    for seed in seeds:
        models_seed_dir = f"{models_dir}/{seed}"
        log_seed_dir = f"{log_dir}/{seed}"
        # create dirs if they do not exist
        Path(models_seed_dir).mkdir(parents=True, exist_ok=True)
        Path(log_seed_dir).mkdir(parents=True, exist_ok=True)

        env.reset(seed=seed)
        if algorithm == "PPO":
            model = PPO(**model_params, env=env, tensorboard_log=log_seed_dir, seed=seed)
        elif algorithm == "DQN":
            model = DQN(**model_params, env=env, tensorboard_log=log_seed_dir, seed=seed)

        # train model
        TIMESTEPS = 10000
        max_iterations = math.ceil(steps / TIMESTEPS)
        for i in range(1, max_iterations+1):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm)
            model.save(f"{models_seed_dir}/{TIMESTEPS*i}")

        # evaluate model
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
        mean_rewards.append(mean_reward)
        reward_stds.append(std_reward)

    print("\nEvaluation results:")
    for mean, std, s in zip(mean_rewards, reward_stds, seeds):
        print(f"Seed {s} | mean reward {round(mean, 2)} | reward std {round(std, 2)}")


def visualize():
    env = em.EnergyManagementEnv()
    env.reset()  # do not seed so we can call it multiple times

    model_path = f"{models_dir}/{steps}.zip"
    if algorithm == "PPO":
        model = PPO.load(model_path, env=env)
    elif algorithm == "DQN":
        model = DQN.load(model_path, env=env)

    episodes = 3

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    time_start = time.time()
    seeds = [0, 1, 42]
    models_dir = f"checkpoints/{args.mode}"
    log_dir = f"tensorboard_logs/{args.mode}"
    algorithm = args.algorithm
    steps = args.steps
    model_params = SB3_PARAMS
    if args.mode == "auto":
        if args.experiment == -1:
            print("Error: Please specify an experiment to run!")
            exit(2)
        experiment = EXPERIMENTS[args.experiment]
        algorithm = experiment["algorithm"]
        steps = experiment["steps"]
        models_dir += f"/{args.experiment}"
        log_dir += f"/{args.experiment}"
        model_params = experiment["sb3_params"]
    elif args.seed:
        seeds = [args.seed]

    if args.action == 'train':
        train()
    else:
        seed = args.seed if args.seed else 0
        models_dir += f"/{seed}"
        visualize()
    time_end = time.time()
    print(f"Took {round(time_end - time_start, 1)}s to run")
