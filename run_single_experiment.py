"""
Run ONE mixture-HBMNL experiment and persist EVERYTHING needed to analyse it
later — including the console/diagnostic output you'd normally watch live.

Invoked by run_all_experiments.py as a subprocess (one process per fit).

Writes into --outdir:
    mcmc_results.pkl    full Goose results object (warmup, tuning, errors, draws)
    posterior_raw.pkl   posterior draws (all params) as numpy arrays
    export.pkl          mu/sigma/std/pvec for marginal-density comparison
    sampling.log        clean Goose engine log (epochs + per-kernel error counts)
    summary.txt         human-readable headline: dims, config, timing, errors
    meta.json           structured config + dimensions + timing + parsed errors
    status.json         {"status": "success"|"failed", ...}  (used for resume)

The orchestrator additionally captures this process's full stdout+stderr into
<outdir>/run.log, so nothing is lost either way; sampling.log and summary.txt
are the *clean* views for fast review.
"""

import argparse
import datetime
import json
import logging
import pathlib
import pickle
import re
import sys
import time
import traceback

import numpy as np


# --------------------------------------------------------------------------- #
# Project root resolution
# --------------------------------------------------------------------------- #
PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _to_numpy(obj):
    """Recursively convert a dict / array of jax arrays to numpy for pickling."""
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    return np.asarray(obj)


def kernel_block_map(has_Z: bool, sampler: str = "nuts") -> dict:
    """
    Map Goose's kernel_NN labels to parameter names, using the block order that
    the respective runner registers kernels in. Lets 'kernel_02' read as 'mu_k'.
    """
    if sampler == "bayesm_gibbs":
        # bayesm sweep order (see src/inference/bayesm_gibbs.py)
        blocks = ["comps(mu_k+sigma_inv_chol_k_latent)", "ind", "pvec_latent"]
    else:
        blocks = ["pvec_latent", "sigma_inv_chol_k_latent", "mu_k"]
    if has_Z:
        blocks.append("Delta")
    blocks.append("beta_i")
    return {f"kernel_{i:02d}": name for i, name in enumerate(blocks)}


def parse_sampling_errors(log_path: pathlib.Path, kmap: dict) -> list:
    """
    Pull the 'Errors per chain for kernel_NN: a / b transitions' lines out of the
    captured Goose log and annotate them with the parameter name. Robust because
    it reads what Goose logged rather than guessing at the results-object API.
    """
    if not log_path.exists():
        return []
    pat = re.compile(r"(kernel_\d+):\s*(\d+)\s*/\s*(\d+)\s*transitions")
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Errors per chain" not in line:
            continue
        m = pat.search(line)
        if not m:
            out.append({"raw": line.strip()})
            continue
        kname = m.group(1)
        out.append({
            "kernel": kname,
            "param": kmap.get(kname, "?"),
            "errors": int(m.group(2)),
            "transitions": int(m.group(3)),
            "epoch_context": line.split(" - ")[-1].strip(),
        })
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Run one mixture-HBMNL experiment.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--k-model", type=int, required=True)
    ap.add_argument("--sampler", required=True,
                    choices=["nuts", "hmc", "iwls", "bayesm_gibbs"])
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--posterior", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--a-delta", type=float, default=0.01)
    ap.add_argument("--a-mu", type=float, default=0.01)
    ap.add_argument("--dirichlet-a", type=float, default=1.0)
    ap.add_argument("--num-integration-steps", type=int, default=10)
    ap.add_argument("--rw-s", type=float, default=None,
                    help="bayesm_gibbs: RW scale (default 2.93/sqrt(n_params)).")
    ap.add_argument("--frac-w", type=float, default=0.1,
                    help="bayesm_gibbs: fractional-likelihood weight w.")
    ap.add_argument("--r-total", type=int, default=42000,
                    help="bayesm_gibbs: total raw Gibbs sweeps "
                         "(matches bayesm Mcmc$R via run_single_bayesm_experiment.R).")
    ap.add_argument("--burn-in", type=int, default=2000,
                    help="bayesm_gibbs: raw sweeps discarded before thinning.")
    ap.add_argument("--thin", type=int, default=4,
                    help="bayesm_gibbs: keep every --thin-th raw draw after burn-in.")
    ap.add_argument("--no-save-raw", action="store_true")
    ap.add_argument("--no-save-results", action="store_true",
                    help="Skip pickling the full mcmc_results object.")
    ap.add_argument("--data-dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status_path   = outdir / "status.json"
    sampling_log  = outdir / "sampling.log"
    summary_path  = outdir / "summary.txt"

    data_dir = (pathlib.Path(args.data_dir) if args.data_dir
                else PROJECT_ROOT / "data" / "simulated" / "mixture")
    data_path = data_dir / f"{args.scenario}.json"

    # ── Capture Goose's logging cleanly into sampling.log ──────────────────
    # Attaching to the "liesel" logger catches all liesel.goose.* child loggers
    # via propagation, regardless of how liesel configures its own handlers.
    fh = logging.FileHandler(sampling_log, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    liesel_logger = logging.getLogger("liesel")
    liesel_logger.setLevel(logging.INFO)
    liesel_logger.addHandler(fh)

    # Collect headline lines for both the console (-> run.log) and summary.txt
    summary_lines = []

    def hlog(msg=""):
        print(msg, flush=True)
        summary_lines.append(msg)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
        from src.mixturemodel import build_mixture_hbmnl_model
        from src.analysis import export_posterior_to_pickle

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "r") as f:
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
        has_Z = choice_data["Z"] is not None
        kmap = kernel_block_map(has_Z, args.sampler)

        # ── Headline block (mirrors the notebook's print cells) ────────────
        hlog("=" * 60)
        hlog(f"Scenario           : {args.scenario}")
        hlog(f"Data file          : {data_path}")
        hlog(f"Units (N)          : {choice_data['n_units']}")
        hlog(f"Parameters (P)     : {P}")
        hlog(f"Demographics (D)   : {choice_data['n_demos']}")
        hlog(f"K in data (K_TRUE) : {K_TRUE}")
        if raw.get("TRUE_PVEC") is not None:
            hlog(f"TRUE_PVEC          : {np.array(raw['TRUE_PVEC']).round(4)}")
        hlog("-" * 60)
        hlog(f"Sampler            : {args.sampler}")
        hlog(f"K_MODEL            : {args.k_model}")
        hlog(f"dirichlet_a        : {args.dirichlet_a}  | a_mu: {args.a_mu} | A_delta: {args.a_delta}")
        if args.sampler == "hmc":
            hlog(f"integration steps  : {args.num_integration_steps}")
        if args.sampler == "bayesm_gibbs":
            hlog(f"rw_s / frac_w      : "
                 f"{args.rw_s if args.rw_s is not None else 'auto (2.93/sqrt(P))'}"
                 f" / {args.frac_w}")
            hlog(f"chains/R/burn/thin : "
                 f"{args.chains} / {args.r_total} / {args.burn_in} / {args.thin}")
        else:
            hlog(f"chains/warmup/post : {args.chains} / {args.warmup} / {args.posterior}")
        hlog(f"kernel -> param    : {kmap}")
        hlog("=" * 60)

        if args.sampler == "bayesm_gibbs":
            # augmented parameterisation with explicit allocations (ind)
            from src.bayesm_mixture_model import build_bayesm_mixture_hbmnl_model
            model = build_bayesm_mixture_hbmnl_model(
                data_dict=choice_data, K=args.k_model,
                A_delta=args.a_delta, a_mu=args.a_mu, dirichlet_a=args.dirichlet_a,
            )
        else:
            model = build_mixture_hbmnl_model(
                data_dict=choice_data, K=args.k_model,
                A_delta=args.a_delta, a_mu=args.a_mu, dirichlet_a=args.dirichlet_a,
            )

        # ── Dispatch sampler ───────────────────────────────────────────────
        if args.sampler == "nuts":
            from src.inference.nuts import run_nuts_inference_mixture_hbmnl
            mcmc_results, posterior_samples = run_nuts_inference_mixture_hbmnl(
                model=model, data_dict=choice_data, K=args.k_model,
                chains=args.chains, warmup=args.warmup,
                posterior=args.posterior, seed=args.seed,
            )
        elif args.sampler == "hmc":
            from src.inference.hmc import run_hmc_inference_mixture_hbmnl
            mcmc_results, posterior_samples = run_hmc_inference_mixture_hbmnl(
                model=model, data_dict=choice_data, K=args.k_model,
                num_integration_steps=args.num_integration_steps,
                chains=args.chains, warmup=args.warmup,
                posterior=args.posterior, seed=args.seed,
            )
        elif args.sampler == "iwls":
            from src.inference.iwls import run_iwls_inference_mixture_hbmnl
            mcmc_results, posterior_samples = run_iwls_inference_mixture_hbmnl(
                model=model, data_dict=choice_data, K=args.k_model,
                chains=args.chains, warmup=args.warmup,
                posterior=args.posterior, seed=args.seed,
            )
        elif args.sampler == "bayesm_gibbs":
            from src.inference.bayesm_gibbs import run_bayesm_gibbs_inference_mixture_hbmnl
            mcmc_results, posterior_samples = run_bayesm_gibbs_inference_mixture_hbmnl(
                model=model, data_dict=choice_data, K=args.k_model,
                chains=args.chains, r_total=args.r_total,
                burn_in=args.burn_in, thin=args.thin, seed=args.seed,
                s=args.rw_s, w=args.frac_w,
            )
        else:
            raise ValueError(f"Unknown sampler: {args.sampler}")

        # ── Persist the two things that matter most ────────────────────────
        if not args.no_save_results:
            try:
                with open(outdir / "mcmc_results.pkl", "wb") as f:
                    pickle.dump(mcmc_results, f)
                hlog(f"Saved mcmc_results.pkl")
            except Exception as e:
                hlog(f"WARNING: could not pickle mcmc_results ({e}); "
                     f"posterior_raw.pkl still holds the draws.")

        if not args.no_save_raw:
            with open(outdir / "posterior_raw.pkl", "wb") as f:
                pickle.dump(_to_numpy(posterior_samples), f)
            hlog(f"Saved posterior_raw.pkl")

        try:
            export_posterior_to_pickle(
                samples=posterior_samples, K=args.k_model, P=P,
                filename="export.pkl", output_dir=str(outdir),
            )
        except Exception as exp_err:
            hlog(f"WARNING export failed: {exp_err}")

        duration = time.time() - t0

        # ── Parse the per-kernel error counts out of the captured Goose log ─
        fh.flush()
        sampling_errors = parse_sampling_errors(sampling_log, kmap)

        # Posterior-epoch errors are the ones that matter most; surface them
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

        if args.sampler == "bayesm_gibbs":
            # mirrors run_single_bayesm_experiment.py's meta.json fields exactly,
            # so both bayesm arms are comparable: warmup<-burn_in, posterior<-
            # retained draws/chain (ground-truthed from the actual output shape).
            n_samples = int(next(iter(posterior_samples.values())).shape[1])
            meta_warmup, meta_posterior = args.burn_in, n_samples
        else:
            meta_warmup, meta_posterior = args.warmup, args.posterior

        meta = {
            "scenario": args.scenario, "data_path": str(data_path),
            "k_model": args.k_model, "k_true": K_TRUE,
            "n_params": P, "n_units": choice_data["n_units"], "n_demos": choice_data["n_demos"],
            "sampler": args.sampler, "chains": args.chains,
            "warmup": meta_warmup, "posterior": meta_posterior, "seed": args.seed,
            "a_delta": args.a_delta, "a_mu": args.a_mu, "dirichlet_a": args.dirichlet_a,
            "num_integration_steps": args.num_integration_steps if args.sampler == "hmc" else None,
            "rw_s": args.rw_s if args.sampler == "bayesm_gibbs" else None,
            "frac_w": args.frac_w if args.sampler == "bayesm_gibbs" else None,
            "r_total": args.r_total if args.sampler == "bayesm_gibbs" else None,
            "burn_in": args.burn_in if args.sampler == "bayesm_gibbs" else None,
            "thin": args.thin if args.sampler == "bayesm_gibbs" else None,
            "n_samples": n_samples if args.sampler == "bayesm_gibbs" else None,
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