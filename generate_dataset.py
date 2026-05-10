import argparse
import os

import numpy as np
import pandas as pd

from cr3bp import integrate_orbit, classify_orbit, make_features


def generate_dataset(n_samples=3000, mu=0.01215, output_path="data/orbits.csv"):
    """
    Generates a labeled CR3BP dataset.

    Labels:
    0 = stable/bounded
    1 = escape
    2 = collision
    """

    rows = []

    for i in range(n_samples):
        # Random initial position
        x0 = np.random.uniform(-1.5, 1.5)
        y0 = np.random.uniform(-1.5, 1.5)

        # Random initial velocity
        vx0 = np.random.uniform(-1.0, 1.0)
        vy0 = np.random.uniform(-1.0, 1.0)

        initial_state = [x0, y0, vx0, vy0]

        try:
            solution = integrate_orbit(
                initial_state=initial_state,
                mu=mu,
                t_max=20.0,
                n_points=1000
            )

            label = classify_orbit(solution, mu)

            features = make_features(x0, y0, vx0, vy0, mu)

            rows.append(features + [label])

        except Exception:
            # If numerical integration fails, skip that sample
            continue

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")

    columns = [
        "mu",
        "x0",
        "y0",
        "vx0",
        "vy0",
        "jacobi_constant",
        "distance_to_primary_1",
        "distance_to_primary_2",
        "label"
    ]

    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nDataset generation complete!")
    print(f"Saved dataset to: {output_path}")
    print("\nClass counts:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_samples",
        type=int,
        default=3000,
        help="Number of random initial conditions to generate"
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=0.01215,
        help="Mass ratio. Earth-Moon is approximately 0.01215"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/orbits.csv",
        help="Where to save the generated dataset"
    )

    args = parser.parse_args()
    

    generate_dataset(
        n_samples=args.n_samples,
        mu=args.mu,
        output_path=args.output_path
    )