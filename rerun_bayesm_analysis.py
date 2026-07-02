"""
Fix scrambled Delta in existing bayesm posterior_raw.pkl files, then re-execute
all bayesm analysis.ipynb notebooks in-place.

Run once after the R-script fix (run_single_bayesm_experiment.R aperm correction):
    uv run python rerun_bayesm_analysis.py

What it does
------------
1. For every bayesm run folder, loads posterior_raw.pkl, applies the index
   correction to Delta (bayesm stores vec(Delta) as column-major of (P, D) but
   the old R code reshaped it as (D, P)), overwrites the pkl.
2. Re-executes analysis.ipynb in-place using jupyter nbconvert.

If you re-run the full bayesm experiments from scratch (which regenerates fresh
pkl files with the fixed R script), you can skip step 1 and just run step 2.
"""

import pathlib
import pickle
import subprocess
import sys

import numpy as np

PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
EXP_ROOT = PROJECT_ROOT / "hbmnl_mixture_experiments"


def fix_delta_pkl(pkl_path: pathlib.Path) -> bool:
    """
    Correct the Delta array in a posterior_raw.pkl produced by the old R code.

    The old aperm used (D, P, S) -> (S, D, P) but bayesm's Deltadraw is
    column-major of (P, D), so it must be (P, D, S) -> (S, D, P).

    Correction: delta_correct[..., d, p] = delta_stored[..., (p+P*d)%D, (p+P*d)//D]
    """
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    if "Delta" not in samples:
        return False

    delta = np.asarray(samples["Delta"])   # (C, S, D, P)
    D, P = delta.shape[-2], delta.shape[-1]

    delta_correct = np.empty_like(delta)
    for d in range(D):
        for p in range(P):
            i = p + P * d
            delta_correct[..., d, p] = delta[..., i % D, i // D]

    if np.allclose(delta, delta_correct):
        print(f"  [skip]  {pkl_path.relative_to(PROJECT_ROOT)}  (already correct)")
        return False

    samples["Delta"] = delta_correct
    with open(pkl_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"  [fixed] {pkl_path.relative_to(PROJECT_ROOT)}")
    return True


def execute_notebook(nb_path: pathlib.Path) -> bool:
    """Run a notebook in-place with jupyter nbconvert --execute."""
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(nb_path),
        ],
        cwd=str(nb_path.parent),
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    status = "OK " if ok else "ERR"
    print(f"  [{status}] {nb_path.relative_to(PROJECT_ROOT)}")
    if not ok:
        for line in result.stderr.splitlines()[-10:]:
            print(f"         {line}")
    return ok


def main():
    bayesm_pkls = sorted(EXP_ROOT.glob("**/bayesm/**/results/posterior_raw.pkl"))
    bayesm_nbs  = sorted(EXP_ROOT.glob("**/bayesm/**/analysis.ipynb"))

    if not bayesm_pkls:
        print("No bayesm posterior_raw.pkl files found.")
        return

    print(f"\n=== Step 1: Fix Delta in {len(bayesm_pkls)} pkl file(s) ===")
    for p in bayesm_pkls:
        fix_delta_pkl(p)

    print(f"\n=== Step 2: Re-execute {len(bayesm_nbs)} notebook(s) ===")
    results = [execute_notebook(nb) for nb in bayesm_nbs]
    n_ok = sum(results)
    print(f"\nDone: {n_ok}/{len(bayesm_nbs)} notebooks succeeded.")


if __name__ == "__main__":
    main()
