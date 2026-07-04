"""
Copy notebook templates into every folder that should have them.

`analysis.ipynb` and `label_switching.ipynb` are per-run notebooks: one per
<sampler>/<run>/ folder (found via <run>/results/posterior_raw.pkl). Each is
self-configuring - it reads meta.json at runtime to locate its own artifacts.

`marginal_comparison.ipynb` and `full_marginal_comparison.ipynb` are per-<k>_comp
notebooks: each contrasts the sampler runs that sit side by side in one <k>_comp
folder, so a single copy is placed at that level rather than per-run.
`marginal_comparison.ipynb` clips the grid to the fitted models' live-component
support; `full_marginal_comparison.ipynb` runs the comparison twice per grid it
builds - once on the full, unbounded envelope over every component, and once on
a Chebyshev-filtered window (mean +/- 5 std, >=96% density guarantee) - to keep
sampler-outlier tails from dominating the unbounded pass.

Usage
    uv run python distribute_notebooks.py                              # all three, copy where missing
    uv run python distribute_notebooks.py --which analysis             # just one
    uv run python distribute_notebooks.py --which analysis,label_switching
    uv run python distribute_notebooks.py --force                      # overwrite existing
    uv run python distribute_notebooks.py --dry-run                    # list targets only
    uv run python distribute_notebooks.py --which analysis --name custom.ipynb
"""

import argparse
import pathlib
import shutil


PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
DEFAULT_EXP_ROOT = "hbmnl_mixture_experiments"

# key -> (template filename, output filename, target level)
#   "run"    : <sampler>/<run>/ folder (parent.parent of results/posterior_raw.pkl)
#   "k_comp" : <chains>/<k>_comp/ folder (parents[3] of results/posterior_raw.pkl),
#              shared by every sampler folder at that comp level
NOTEBOOKS = {
    "analysis":            ("analysis_template.ipynb",            "analysis.ipynb",            "run"),
    "standard_analysis":   ("standard_analysis_template.ipynb",    "analysis.ipynb",            "run"),
    "label_switching":     ("label_switching_template.ipynb",      "label_switching.ipynb",      "run"),
    "marginal_comparison": ("marginal_comparison_template.ipynb",  "marginal_comparison.ipynb",  "k_comp"),
    "full_marginal_comparison": (
        "full_marginal_comparison_template.ipynb", "full_marginal_comparison.ipynb", "k_comp",
    ),
    "standard_model_comparison": (
        "standard_model_comparison_template.ipynb", "model_comparison.ipynb", "k_comp",
    ),
}


def find_targets(exp_root, level):
    pkls = list(exp_root.rglob("posterior_raw.pkl"))
    if level == "run":
        return sorted({p.parent.parent for p in pkls})
    if level == "k_comp":
        return sorted({p.parents[3] for p in pkls})
    raise ValueError(level)


def distribute(exp_root, key, template_name, output_name, level, name_override, force, dry_run):
    template = PROJECT_ROOT / template_name
    if not template.exists():
        print(f"[{key}] SKIP: template not found: {template.relative_to(PROJECT_ROOT)}")
        return 0, 0

    targets = find_targets(exp_root, level)
    if not targets:
        print(f"[{key}] no target folders found under {exp_root} (no posterior_raw.pkl files).")
        return 0, 0

    name = name_override or output_name
    print(f"[{key}] template: {template.relative_to(PROJECT_ROOT)}  ->  {name}  "
          f"({len(targets)} target folder(s))")

    copied = skipped = 0
    for d in targets:
        dest = d / name
        rel = dest.relative_to(PROJECT_ROOT)
        if dry_run:
            mark = "exists" if dest.exists() else "new"
            print(f"  [{mark:>6}] {rel}")
            continue
        if dest.exists() and not force:
            print(f"  SKIP (exists)  {rel}")
            skipped += 1
            continue
        shutil.copyfile(template, dest)
        print(f"  WROTE          {rel}")
        copied += 1
    return copied, skipped


def main():
    ap = argparse.ArgumentParser(description="Copy notebook templates into every folder that should have them.")
    ap.add_argument("--which", default=",".join(NOTEBOOKS),
                    help=f"Comma-separated notebook keys to distribute "
                         f"(default: all - {', '.join(NOTEBOOKS)}).")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing notebook.")
    ap.add_argument("--dry-run", action="store_true", help="List target folders and exit.")
    ap.add_argument("--name", default=None,
                    help="Filename to write instead of the default. Only valid with a single --which key.")
    ap.add_argument("--exp-root", default=DEFAULT_EXP_ROOT,
                    help=f"Experiments tree to distribute into, relative to the repo root "
                         f"(default: {DEFAULT_EXP_ROOT}; e.g. hbmnl_normal_experiments).")
    args = ap.parse_args()

    keys = [k.strip() for k in args.which.split(",") if k.strip()]
    unknown = set(keys) - set(NOTEBOOKS)
    if unknown:
        ap.error(f"Unknown notebook key(s): {sorted(unknown)}; valid: {sorted(NOTEBOOKS)}")
    if args.name and len(keys) != 1:
        ap.error("--name requires exactly one --which key.")

    exp_root = PROJECT_ROOT / args.exp_root
    if not exp_root.exists():
        ap.error(f"--exp-root does not exist: {exp_root}")

    total_copied = total_skipped = 0
    for key in keys:
        template_name, output_name, level = NOTEBOOKS[key]
        copied, skipped = distribute(
            exp_root, key, template_name, output_name, level,
            args.name, args.force, args.dry_run,
        )
        total_copied += copied
        total_skipped += skipped

    if not args.dry_run:
        print(f"\nDone. {total_copied} written, {total_skipped} skipped "
              f"(use --force to overwrite skipped).")


if __name__ == "__main__":
    main()
