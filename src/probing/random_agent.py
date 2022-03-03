import src.environment.energy_management as em
import gym


def run_episode(env: gym.Env):
    """Use random policy on environment."""

    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())


def main():
    env = em.EnergyManagementEnv()

    while True:
        env.reset()
        run_episode(env)
        env.render()


if __name__ == "__main__":
    main()
