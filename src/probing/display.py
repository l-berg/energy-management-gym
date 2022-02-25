import src.environment.energy_management as em


def single_action_run():
    env = em.EnergyManagementEnv()
    env.reset()

    done = False
    while not done:
        _, _, done, _ = env.step(0)

    env.render()


def main():
    while True:
        single_action_run()


if __name__ == "__main__":
    main()