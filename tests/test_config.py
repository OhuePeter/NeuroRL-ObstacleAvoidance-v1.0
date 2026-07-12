from src.utils.config import ConfigLoader


def main():

    env = ConfigLoader.load_environment()

    perturb = ConfigLoader.load_perturbation()

    print("=" * 50)

    print("Environment Configuration")

    print("=" * 50)

    print(env)

    print()

    print("=" * 50)

    print("Perturbation Configuration")

    print("=" * 50)

    print(perturb)


if __name__ == "__main__":
    main()