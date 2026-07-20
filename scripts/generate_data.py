import argparse
import os
import pathlib
import sys

PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dgp import generate_mixture_simulated_data, save_to_json
from hbmnl_mixture_experiments.experiment_configs import SCENARIOS

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "simulated", "mixture")


def generate_scenario(name: str, params: dict) -> None:
    print(f"\n{'─' * 55}")
    print(f"  Scenario : {name}")
    print(f"  K        : {params['n_components']}")
    print(f"  n_units  : {params['n_units']}")
    print(f"  n_obs    : {params['n_obs']}")
    print(f"  pvec     : {params.get('custom_pvec')}")
    print(f"{'─' * 55}")

    data = generate_mixture_simulated_data(**params)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    save_to_json(data, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate HBMNL mixture simulation datasets."
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Name of a single scenario to generate. Omit to generate all."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available scenario names and exit."
    )
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, params in SCENARIOS.items():
            print(
                f"  {name:<25} "
                f"K={params['n_components']}, "
                f"n_units={params['n_units']}, "
                f"n_obs={params['n_obs']}"
            )
        return

    if args.scenario:
        if args.scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{args.scenario}'.\n"
                f"Run with --list to see available scenarios."
            )
        targets = {args.scenario: SCENARIOS[args.scenario]}
    else:
        targets = SCENARIOS

    print(f"\nOutput directory : {OUTPUT_DIR}")
    print(f"Scenarios to run : {len(targets)}")

    for name, params in targets.items():
        generate_scenario(name, params)

    print(f"\nDone - {len(targets)} dataset(s) written to data/simulated/mixture/")


if __name__ == "__main__":
    main()