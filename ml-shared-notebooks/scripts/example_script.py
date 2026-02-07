from pathlib import Path
import numpy as np
import pandas as pd


def main(seed: int = 7, n: int = 400) -> None:
    rng = np.random.default_rng(seed)

    # scripts/make_example_data.py
    # repo_root = .../ml-shared-notebooks
    # data is now at .../data
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root.parent / "data" / "raw"

    data_dir.mkdir(parents=True, exist_ok=True)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.8, n)

    # binary target
    logit = 1.2 * x1 - 0.9 * x2 + noise
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    out_path = data_dir / "example.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
