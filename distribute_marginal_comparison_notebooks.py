"""
Copy the marginal-comparison notebook template into every <chains>/<k>_comp/ folder.

Unlike the per-run analysis / label_switching notebooks, the marginal comparison
contrasts the NUTS, HMC and bayesm runs that sit side by side, so ONE notebook is
placed at the <k>_comp level (the parent of the sampler folders). It is found by
descending to every `<sampler>/<run>/results/posterior_raw.pkl` and taking the
<k>_comp folder two levels above the sampler. Purely additive.

Usage
    uv run python distribute_marginal_comparison_notebooks.py            # copy where missing
    uv run python distribute_marginal_comparison_notebooks.py --force    # overwrite existing
    uv run python distribute_marginal_comparison_notebooks.py --dry-run  # list targets only
"""

import argparse
import pathlib
import shutil
import sys


PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
TEMPLATE = PROJECT_ROOT / "marginal_comparison_template.ipynb"
EXP_ROOT = PROJECT_ROOT / "hbmnl_mixture_experiments"
NOTEBOOK_NAME = "marginal_comparison.ipynb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite an existing notebook.")
    ap.add_argument("--dry-run", action="store_true", help="List target folders and exit.")
    ap.add_argument("--name", default=NOTEBOOK_NAME, help="Filename to write in each x_comp folder.")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        sys.exit(f"Template not found: {TEMPLATE}\n"
                 f"Build marginal_comparison_template.ipynb at the repo root first.")

    # <chains>/<k>_comp/<sampler>/<run>/results/posterior_raw.pkl -> x_comp = parents[3]
    xcomp_dirs = sorted({p.parents[3] for p in EXP_ROOT.rglob("posterior_raw.pkl")})
    if not xcomp_dirs:
        sys.exit(f"No <k>_comp folders found under {EXP_ROOT} (no posterior_raw.pkl files).")

    print(f"Template : {TEMPLATE.relative_to(PROJECT_ROOT)}")
    print(f"x_comp folders found : {len(xcomp_dirs)}\n")

    copied = skipped = 0
    for d in xcomp_dirs:
        dest = d / args.name
        rel = dest.relative_to(PROJECT_ROOT)
        if args.dry_run:
            mark = "exists" if dest.exists() else "new"
            print(f"  [{mark:>6}] {rel}")
            continue
        if dest.exists() and not args.force:
            print(f"  SKIP (exists)  {rel}")
            skipped += 1
            continue
        shutil.copyfile(TEMPLATE, dest)
        print(f"  WROTE          {rel}")
        copied += 1

    if not args.dry_run:
        print(f"\nDone. {copied} written, {skipped} skipped "
              f"(use --force to overwrite skipped).")


if __name__ == "__main__":
    main()
