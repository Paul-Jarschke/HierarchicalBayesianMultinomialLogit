import argparse
import json
import pathlib
import subprocess
import sys
import time


PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
DEFAULT_EXP_ROOT = "hbmnl_mixture_experiments"
NOTEBOOK_NAME    = "analysis.ipynb"


def find_notebooks(exp_root, filter_str=None, name=NOTEBOOK_NAME):
    """Return sorted list of notebooks named <name> anywhere under exp_root.

    Works for the per-run notebooks (analysis.ipynb, label_switching.ipynb, which
    live beside a results/ dir) and the x_comp-level marginal_comparison.ipynb
    alike - each is executed with its own folder as cwd so it self-resolves."""
    notebooks = sorted(exp_root.rglob(name))
    if filter_str:
        filter_norm = filter_str.replace("\\", "/")
        notebooks = [nb for nb in notebooks if filter_norm in str(nb).replace("\\", "/")]
    return notebooks


def is_executed(notebook: pathlib.Path) -> bool:
    """Return True if any code cell has a non-null execution_count."""
    try:
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        return any(
            cell.get("execution_count") is not None
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
    except Exception:
        return False


def execute_notebook(notebook, timeout):
    """Run a single notebook in-place. Returns (success, elapsed_seconds, stderr)."""
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=notebook.parent,   # so _resolve_run_dir() fallback points here
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed, result.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="List targets without executing.")
    ap.add_argument("--force",   action="store_true", help="Re-run already-executed notebooks.")
    ap.add_argument("--timeout", type=int, default=600, help="Seconds allowed per notebook (default 600).")
    ap.add_argument("--filter",  default=None, help="Only execute notebooks whose path contains this string.")
    ap.add_argument("--name",    default=NOTEBOOK_NAME,
                    help=f"Notebook filename to execute in each run folder (default: {NOTEBOOK_NAME}; "
                         f"e.g. label_switching.ipynb).")
    ap.add_argument("--exp-root", default=DEFAULT_EXP_ROOT,
                    help=f"Experiments tree to search, relative to the repo root "
                         f"(default: {DEFAULT_EXP_ROOT}; e.g. hbmnl_normal_experiments).")
    args = ap.parse_args()

    exp_root = PROJECT_ROOT / args.exp_root
    notebooks = find_notebooks(exp_root, args.filter, args.name)

    if not notebooks:
        sys.exit(f"No {args.name} files found under {exp_root}.")

    print(f"Notebooks found : {len(notebooks)}")
    if args.filter:
        print(f"Filter          : {args.filter}")
    print(f"Timeout / nb    : {args.timeout}s")
    print(f"Force re-run    : {args.force}\n")

    if args.dry_run:
        for nb in notebooks:
            mark = "executed" if is_executed(nb) else "pending"
            print(f"  [{mark:>8}] {nb.relative_to(PROJECT_ROOT)}")
        return

    succeeded, failed, skipped = [], [], []

    for i, nb in enumerate(notebooks, 1):
        rel = nb.relative_to(PROJECT_ROOT)
        if not args.force and is_executed(nb):
            print(f"[{i}/{len(notebooks)}] SKIP (already executed)  {rel}")
            skipped.append(rel)
            continue
        print(f"[{i}/{len(notebooks)}] {rel} ...", end=" ", flush=True)
        ok, elapsed, stderr = execute_notebook(nb, args.timeout)
        if ok:
            print(f"OK  ({elapsed:.0f}s)")
            succeeded.append(rel)
        else:
            print(f"FAILED  ({elapsed:.0f}s)")
            tail = "\n".join(stderr.strip().splitlines()[-6:])
            print(f"        {tail}\n")
            failed.append(rel)

    print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped.")
    if failed:
        print("\nFailed notebooks:")
        for nb in failed:
            print(f"  {nb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
