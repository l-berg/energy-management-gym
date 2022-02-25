import src.environment.energy_management as em


def run_episode():
    """Use random policy on environment."""
    env = em.EnergyManagementEnv()
    env.reset()

    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())

    env.render()


def main():
    while True:
        run_episode()


if __name__ == "__main__":
    main()
