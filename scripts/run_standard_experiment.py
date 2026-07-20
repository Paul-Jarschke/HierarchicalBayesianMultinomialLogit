import argparse
import datetime
import json
import logging
import pathlib
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np


PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "simulated" / "mixture" / "standard.json"
EXP_ROOT  = PROJECT_ROOT / "hbmnl_normal_experiments"

from run_single_experiment import _to_numpy, parse_sampling_errors  # noqa: E402

# Goose kernel_NN -> parameter, in the order src.standardmodel registers kernels.
KERNEL_MAP = {
    "kernel_00": "sigma_inv_chol_latent",
    "kernel_01": "mu",
    "kernel_02": "Delta",
    "kernel_03": "beta_i",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Run one standard (no-mixture) HBMNL experiment.")
    ap.add_argument("--sampler", required=True, choices=["nuts", "hmc", "bayesm"])
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--posterior", type=int, default=10000)
    ap.add_argument("--num-integration-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--a-delta", type=float, default=0.01,
                    help="Prior precision scaling for Delta (bayesm's Ad).")
    ap.add_argument("--a-mu", type=float, default=0.01,
                    help="Prior precision scaling for mu (bayesm's Amu). "
                         "Rossi §5.4 'standard diffuse': 0.01.")
    ap.add_argument("--r-total", type=int, default=42000,
                    help="bayesm: total raw Gibbs sweeps.")
    ap.add_argument("--burn-in", type=int, default=2000,
                    help="bayesm: raw sweeps discarded before thinning.")
    ap.add_argument("--thin", type=int, default=4,
                    help="bayesm: keep every --thin-th raw draw after burn-in.")
    ap.add_argument("--outdir", default=None,
                    help="Default: hbmnl_normal_experiments/<SAMPLER>/standard_seed<seed>/results")
    return ap.parse_args()


def ensure_data():
    """Generate data/simulated/mixture/standard.json on first use (seed 42)."""
    if DATA_PATH.exists():
        return
    from src.dgp import generate_standard_simulated_data, save_to_json
    print(f"Data file missing - generating {DATA_PATH} ...")
    save_to_json(generate_standard_simulated_data(seed=42), str(DATA_PATH))


def _bayesm_to_plain_keys(outdir: pathlib.Path):
    """
    Rewrite the R bridge's posterior_raw.pkl from its mixture-style keys
    (K = 1 component axis) to the plain single-component keys the Liesel
    arms use, so all three arms share one format. The bridge's export.pkl
    (a mixture-comparison artifact keyed by component) is removed for the
    same reason.
    """
    pkl_path = outdir / "posterior_raw.pkl"
    with open(pkl_path, "rb") as f:
        s = pickle.load(f)

    plain = {
        "mu":                    np.asarray(s["mu_k"])[:, :, 0, :],
        "sigma_inv_chol_latent": np.asarray(s["sigma_inv_chol_k_latent"])[:, :, 0, :],
        "beta_i":                np.asarray(s["beta_i"]),
    }
    if "Delta" in s:
        plain["Delta"] = np.asarray(s["Delta"])

    with open(pkl_path, "wb") as f:
        pickle.dump(plain, f)
    (outdir / "export.pkl").unlink(missing_ok=True)
    print("Converted bayesm posterior_raw.pkl to plain single-component keys.")


def run_bayesm(args, outdir: pathlib.Path) -> int:
    """rhierMnlRwMixture with ncomp=1 via the R bridge subprocess."""
    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "run_single_bayesm_experiment.py"),
        "--scenario", "standard",
        "--k-model", "1",
        "--chains", str(args.chains),
        "--seed", str(args.seed),
        "--a-mu", str(args.a_mu),
        "--a-delta", str(args.a_delta),
        "--dirichlet-a", "1.0",
        "--r-total", str(args.r_total),
        "--burn-in", str(args.burn_in),
        "--thin", str(args.thin),
        "--outdir", str(outdir),
    ]
    with open(outdir / "run.log", "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT))
    print(f"bayesm bridge finished with exit code {proc.returncode} "
          f"(artifacts + status.json written by the bridge itself).")
    if proc.returncode == 0:
        _bayesm_to_plain_keys(outdir)
    return proc.returncode


def main():
    args = parse_args()

    outdir = pathlib.Path(args.outdir) if args.outdir else (
        EXP_ROOT / args.sampler.upper() / f"standard_seed{args.seed}" / "results"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_data()

    if args.sampler == "bayesm":
        return run_bayesm(args, outdir)

    status_path  = outdir / "status.json"
    sampling_log = outdir / "sampling.log"
    summary_path = outdir / "summary.txt"

    fh = logging.FileHandler(sampling_log, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    liesel_logger = logging.getLogger("liesel")
    liesel_logger.setLevel(logging.INFO)
    liesel_logger.addHandler(fh)

    summary_lines = []

    def hlog(msg=""):
        print(msg, flush=True)
        summary_lines.append(msg)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    try:
        import jax.numpy as jnp
        from src.standardmodel import (
            build_standard_hbmnl_model,
            run_nuts_inference_standard_hbmnl,
            run_hmc_inference_standard_hbmnl,
        )

        with open(DATA_PATH) as f:
            raw = json.load(f)
        K_TRUE = int(raw["K"])
        P = int(raw["n_params"])

        choice_data = {
            "X":        jnp.array(raw["X"]),
            "y":        jnp.array(raw["y"]),
            "unit_idx": jnp.array(raw["unit_idx"]),
            "Z":        jnp.array(raw["Z"]) if raw.get("Z") is not None else None,
            "n_params": P,
            "n_units":  int(raw["n_units"]),
            "n_demos":  int(raw.get("n_demos", 0)),
        }
        kmap = dict(KERNEL_MAP)
        if choice_data["Z"] is None:
            # no Delta kernel; beta_i moves up one slot
            kmap = {"kernel_00": "sigma_inv_chol_latent",
                    "kernel_01": "mu", "kernel_02": "beta_i"}

        hlog("=" * 60)
        hlog(f"Model              : standard HBMNL (one normal component, Rossi §5.4)")
        hlog(f"Data file          : {DATA_PATH}")
        hlog(f"Units (N)          : {choice_data['n_units']}")
        hlog(f"Parameters (P)     : {P}")
        hlog(f"Demographics (D)   : {choice_data['n_demos']}")
        hlog(f"K in data (K_TRUE) : {K_TRUE}")
        hlog("-" * 60)
        hlog(f"Sampler            : {args.sampler}")
        hlog(f"a_mu / A_delta     : {args.a_mu} / {args.a_delta}")
        if args.sampler == "hmc":
            hlog(f"integration steps  : {args.num_integration_steps}")
        hlog(f"chains/warmup/post : {args.chains} / {args.warmup} / {args.posterior}")
        hlog(f"kernel -> param    : {kmap}")
        hlog("=" * 60)

        model = build_standard_hbmnl_model(
            choice_data, A_delta=args.a_delta, a_mu=args.a_mu,
        )

        if args.sampler == "nuts":
            mcmc_results, posterior_samples = run_nuts_inference_standard_hbmnl(
                model, chains=args.chains, warmup=args.warmup,
                posterior=args.posterior, seed=args.seed,
            )
        else:
            mcmc_results, posterior_samples = run_hmc_inference_standard_hbmnl(
                model, num_integration_steps=args.num_integration_steps,
                chains=args.chains, warmup=args.warmup,
                posterior=args.posterior, seed=args.seed,
            )

        try:
            with open(outdir / "mcmc_results.pkl", "wb") as f:
                pickle.dump(mcmc_results, f)
            hlog("Saved mcmc_results.pkl")
        except Exception as e:
            hlog(f"WARNING: could not pickle mcmc_results ({e}); "
                 f"posterior_raw.pkl still holds the draws.")

        with open(outdir / "posterior_raw.pkl", "wb") as f:
            pickle.dump(_to_numpy(posterior_samples), f)
        hlog("Saved posterior_raw.pkl")

        duration = time.time() - t0

        fh.flush()
        sampling_errors = parse_sampling_errors(sampling_log, kmap)

        hlog("-" * 60)
        hlog("Per-kernel error counts (all epochs, from sampling.log):")
        if sampling_errors:
            for e in sampling_errors:
                if "raw" in e:
                    hlog(f"   {e['raw']}")
                else:
                    hlog(f"   {e['param']:<26} {e['errors']:>5} / {e['transitions']:<6} "
                         f"[{e['epoch_context']}]")
        else:
            hlog("   (none logged)")
        hlog("-" * 60)
        hlog(f"DONE in {datetime.timedelta(seconds=int(duration))}")

        meta = {
            "model": "standard_hbmnl", "scenario": "standard",
            "data_path": str(DATA_PATH),
            "k_model": 1, "k_true": K_TRUE,
            "n_params": P, "n_units": choice_data["n_units"],
            "n_demos": choice_data["n_demos"],
            "sampler": args.sampler, "chains": args.chains,
            "warmup": args.warmup, "posterior": args.posterior, "seed": args.seed,
            "a_delta": args.a_delta, "a_mu": args.a_mu,
            "num_integration_steps": (args.num_integration_steps
                                      if args.sampler == "hmc" else None),
            "kernel_map": kmap,
            "sampling_errors": sampling_errors,
            "started_at": started_at, "duration_s": round(duration, 1),
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        with open(status_path, "w") as f:
            json.dump({"status": "success", "duration_s": round(duration, 1),
                       "finished_at": datetime.datetime.now().isoformat(timespec="seconds")},
                      f, indent=2)
        return 0

    except Exception:
        duration = time.time() - t0
        err = traceback.format_exc()
        hlog(err)
        try:
            summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        except Exception:
            pass
        with open(status_path, "w") as f:
            json.dump({"status": "failed", "error": err, "duration_s": round(duration, 1),
                       "finished_at": datetime.datetime.now().isoformat(timespec="seconds")},
                      f, indent=2)
        return 1

    finally:
        fh.flush()
        fh.close()
        liesel_logger.removeHandler(fh)


if __name__ == "__main__":
    sys.exit(main())
