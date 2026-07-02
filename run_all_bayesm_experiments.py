"""
Orchestrate a batch of bayesm mixture-HBMNL experiments - the R counterpart of
run_all_experiments.py.

Runs every experiment in the grid as a SEPARATE SUBPROCESS (one process per fit)
via run_single_bayesm_experiment.py, which in turn drives the R sampler. Same
guarantees as the Liesel orchestrator:

  * Resumable  : experiments whose outdir already has status=="success" are skipped.
  * Robust     : each subprocess has a timeout; failures are logged, batch continues.
  * Auditable  : a master log + manifest.csv summarise every run.

Output lands in the SAME tree as the Liesel runs, with a BAYESM/ folder beside
NUTS/ and HMC/:
    hbmnl_mixture_experiments/<chains>/<k>_comp/BAYESM/<run>/results/

Usage
    uv run python run_all_bayesm_experiments.py                 # run the grid
    uv run python run_all_bayesm_experiments.py --dry-run       # print plan only
    uv run python run_all_bayesm_experiments.py --force         # re-run even if done
    uv run python run_all_bayesm_experiments.py --strategy known
"""

import argparse
import csv
import datetime
import json
import pathlib
import subprocess
import sys
import time


PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
SINGLE_RUNNER = PROJECT_ROOT / "run_single_bayesm_experiment.py"
RESULTS_ROOT  = PROJECT_ROOT / "hbmnl_mixture_experiments"
LOG_DIR       = PROJECT_ROOT / "batch_logs"


# --------------------------------------------------------------------------- #
# Experiment grid (mirrors run_all_experiments.py; sampler is fixed to bayesm)
# --------------------------------------------------------------------------- #
try:
    from hbmnl_mixture_experiments.experiment_configs import SCENARIOS as _CFG
    SCENARIOS = [(name, int(p["n_components"])) for name, p in _CFG.items()]
except Exception:
    SCENARIOS = [
        ("1comp",       1),
        ("2comp_equal", 2),
        ("3comp_equal", 3),
        ("5comp_equal", 5),
    ]

CHAINS_GRID = [1, 2, 4]         # match the Liesel CHAINS_GRID

# bayesm MCMC length. The R side keeps ALL raw draws, discards the first BURN_IN
# (warmup), THEN thins by THIN - so retained/chain = (R_TOTAL - BURN_IN) / THIN.
# (42000 - 2000) / 4 = 10000 kept draws, matching the 10000 posterior draws the
# Liesel NUTS/HMC chains retain. Rossi advises a long chain (>20k); THIN=4 lightly
# thins the autocorrelated RW-Metropolis sampler.
R_TOTAL   = 42000
BURN_IN   = 2000
THIN      = 4
SEED      = 42

# Priors - matched to the Liesel model / run_all_experiments.py
A_DELTA     = 0.01
A_MU        = 0.0625            # 1/16, per Rossi p.150 for standardised X
DIRICHLET_A = 1.0

# Per-experiment wall-clock cap; a stuck fit is killed so the batch continues.
TIMEOUT_S = 6 * 60 * 60        # 6 hours


def resolve_k_model(k_true: int, strategy: str) -> int:
    if strategy == "fixed5":
        return 5
    if strategy == "known":
        return k_true
    raise ValueError(f"Unknown strategy: {strategy}")


def chains_label(chains: int) -> str:
    return "1_chain" if chains == 1 else f"{chains}_chains"


def build_grid(strategy: str):
    grid = []
    for chains in CHAINS_GRID:
        for scenario, k_true in SCENARIOS:
            k_model = resolve_k_model(k_true, strategy)
            outdir = (RESULTS_ROOT / chains_label(chains) / f"{k_true}_comp"
                      / "BAYESM" / f"{scenario}_K{k_model}_seed{SEED}"
                      / "results")
            grid.append({
                "scenario": scenario, "k_true": k_true, "k_model": k_model,
                "chains": chains, "outdir": outdir,
            })
    # All 1-chain runs before 2-chain (stable sort keeps scenario order).
    grid.sort(key=lambda e: e["chains"])
    return grid


def is_done(outdir: pathlib.Path) -> bool:
    status_file = outdir / "status.json"
    if not status_file.exists():
        return False
    try:
        with open(status_file) as f:
            return json.load(f).get("status") == "success"
    except Exception:
        return False


def run_one(exp: dict, log_path: pathlib.Path) -> tuple[str, float]:
    """Launch the single bayesm runner as a subprocess. Returns (status, secs)."""
    cmd = [
        sys.executable, "-u", str(SINGLE_RUNNER),
        "--scenario", exp["scenario"],
        "--k-model", str(exp["k_model"]),
        "--chains", str(exp["chains"]),
        "--seed", str(SEED),
        "--a-delta", str(A_DELTA),
        "--a-mu", str(A_MU),
        "--dirichlet-a", str(DIRICHLET_A),
        "--r-total", str(R_TOTAL),
        "--burn-in", str(BURN_IN),
        "--thin", str(THIN),
        "--outdir", str(exp["outdir"]),
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT), timeout=TIMEOUT_S,
            )
            status = "success" if proc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
    return status, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["fixed5", "known"], default="fixed5",
                    help="fixed5: fit K_MODEL=5 everywhere; known: fit K_MODEL=K_TRUE")
    ap.add_argument("--force", action="store_true", help="Re-run even completed experiments.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = ap.parse_args()

    grid = build_grid(args.strategy)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = LOG_DIR / f"bayesm_batch_{stamp}.log"
    manifest   = LOG_DIR / f"bayesm_manifest_{stamp}.csv"

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with open(master_log, "a") as f:
            f.write(line + "\n")

    log(f"[bayesm] Strategy={args.strategy}  experiments={len(grid)}  timeout/exp={TIMEOUT_S}s")
    log(f"Master log : {master_log}")
    log(f"Manifest   : {manifest}")

    if args.dry_run:
        for i, e in enumerate(grid, 1):
            done = "DONE" if is_done(e["outdir"]) else "todo"
            print(f"  [{i:>2}/{len(grid)}] {done}  bayesm  "
                  f"chains={e['chains']}  {e['scenario']:<13} "
                  f"K_MODEL={e['k_model']}  -> {e['outdir'].relative_to(PROJECT_ROOT)}")
        return

    with open(manifest, "w", newline="") as f:
        csv.writer(f).writerow(
            ["idx", "scenario", "k_true", "k_model", "sampler", "chains",
             "status", "duration_s", "outdir"]
        )

    overall_t0 = time.time()
    counts = {"success": 0, "failed": 0, "timeout": 0, "skipped": 0}

    for i, exp in enumerate(grid, 1):
        tag = (f"[{i}/{len(grid)}] bayesm chains={exp['chains']} "
               f"{exp['scenario']} K_MODEL={exp['k_model']}")

        if not args.force and is_done(exp["outdir"]):
            log(f"SKIP (already done)  {tag}")
            counts["skipped"] += 1
            with open(manifest, "a", newline="") as f:
                csv.writer(f).writerow(
                    [i, exp["scenario"], exp["k_true"], exp["k_model"],
                     "bayesm", exp["chains"], "skipped", 0, exp["outdir"]]
                )
            continue

        exp["outdir"].mkdir(parents=True, exist_ok=True)
        log_path = exp["outdir"] / "run.log"
        log(f"START  {tag}")
        try:
            status, secs = run_one(exp, log_path)
        except KeyboardInterrupt:
            log("KeyboardInterrupt - stopping batch.")
            break
        counts[status] = counts.get(status, 0) + 1
        log(f"END    {tag}  -> {status}  ({datetime.timedelta(seconds=int(secs))})")

        with open(manifest, "a", newline="") as f:
            csv.writer(f).writerow(
                [i, exp["scenario"], exp["k_true"], exp["k_model"],
                 "bayesm", exp["chains"], status, round(secs, 1), exp["outdir"]]
            )

    total = datetime.timedelta(seconds=int(time.time() - overall_t0))
    log(f"BAYESM BATCH COMPLETE in {total}  | "
        f"success={counts['success']} failed={counts['failed']} "
        f"timeout={counts['timeout']} skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
