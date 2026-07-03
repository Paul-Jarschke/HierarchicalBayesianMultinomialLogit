"""
Run ONE mixture-HBMNL fit with bayesm and persist the SAME artifacts the Liesel
runner does, so the identical analysis notebook + src/analysis.py work unchanged.

This is the bayesm counterpart of run_single_experiment.py. It:
  1. invokes run_single_bayesm_experiment.R as a subprocess (pure sampling),
     which dumps raw float64 draws + dims.json/meta_r.json into <outdir>/_bayesm_raw/,
  2. converts those raw draws into the canonical posterior_raw.pkl dict, with the
     EXACT keys/shapes the Liesel pipeline produces:
         mu_k                     (chains, draws, K, P)
         sigma_inv_chol_k_latent  (chains, draws, K, P(P+1)/2)  - TFP FillScaleTriL
         pvec                     (chains, draws, K)
         Delta                    (chains, draws, D, P)         - only if demographics
         beta_i                   (chains, draws, N, P)
  3. writes export.pkl (via the shared export_posterior_to_pickle), meta.json,
     status.json, summary.txt and sampling.log - same names the notebook expects.

The crucial detail: bayesm samples Sigma_k directly (Gibbs), with no Cholesky-of-
precision latent. We rebuild that latent so the SAME analysis code can invert it
back to Sigma. R exports the precision Sigma^{-1} = rooti @ rooti.T (bayesm's own
definition); here we take its lower Cholesky L (L L.T = precision) and map it
through FillScaleTriL().inverse - the exact representation the Liesel model stores.

Invoked by run_all_experiments.py (--samplers bayesm) as a subprocess (one process per fit).
"""

import argparse
import datetime
import json
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


# --------------------------------------------------------------------------- #
# Project root resolution (same idiom as run_single_experiment.py)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

R_SCRIPT = PROJECT_ROOT / "run_single_bayesm_experiment.R"

# Rscript location: env override wins, else the known Windows install, else PATH.
import os
_DEFAULT_RSCRIPT = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
RSCRIPT = os.environ.get("BAYESM_RSCRIPT") or (
    _DEFAULT_RSCRIPT if pathlib.Path(_DEFAULT_RSCRIPT).exists() else "Rscript"
)


def parse_args():
    ap = argparse.ArgumentParser(description="Run one mixture-HBMNL fit with bayesm.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--k-model", type=int, required=True)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--a-delta", type=float, default=0.01)
    ap.add_argument("--a-mu", type=float, default=0.0625)
    ap.add_argument("--dirichlet-a", type=float, default=1.0)
    # bayesm MCMC controls. The R side keeps ALL raw draws, discards the first
    # burn_in (warmup) iterations, THEN thins by `thin` - in that order, in raw
    # iteration units. Retained/chain = len(range(burn_in+1, r_total, thin)):
    #   (42000 - 2000) / 4 = 10000 retained draws, matching the Liesel chains.
    ap.add_argument("--r-total", type=int, default=42000)
    ap.add_argument("--burn-in", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--nprint", type=int, default=2000)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--keep-raw", action="store_true",
                    help="Keep the large _bayesm_raw/*.bin files (default: delete after conversion).")
    return ap.parse_args()


def _read_bin(path, shape):
    """Read an R column-major float64 dump and reshape with Fortran order."""
    arr = np.fromfile(path, dtype="<f8")
    return arr.reshape(tuple(shape), order="F")


def _precision_to_latent(prec):
    """
    prec : (C, S, K, P, P) precision matrices (Sigma^{-1}).
    Returns the TFP FillScaleTriL latent (C, S, K, P(P+1)/2) such that
    recover_covariance_matrices() / _sigma_from_latent() round-trip back to Sigma.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax.bijectors as tfb

    # Symmetrise to kill tiny asymmetry, then lower-Cholesky: L L^T = precision.
    prec = 0.5 * (prec + np.swapaxes(prec, -1, -2))
    L = np.linalg.cholesky(prec)                       # lower-triangular, positive diag
    latent = np.asarray(tfb.FillScaleTriL().inverse(jnp.asarray(L)))
    return latent


def main():
    args = parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "_bayesm_raw"

    status_path  = outdir / "status.json"
    summary_path = outdir / "summary.txt"
    sampling_log = outdir / "sampling.log"

    data_dir = (pathlib.Path(args.data_dir) if args.data_dir
                else PROJECT_ROOT / "data" / "simulated" / "mixture")
    data_path = data_dir / f"{args.scenario}.json"

    summary_lines = []

    def hlog(msg=""):
        print(msg, flush=True)
        summary_lines.append(msg)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path) as f:
            raw = json.load(f)
        K_TRUE = int(raw["K"])
        P = int(raw["n_params"])
        N = int(raw["n_units"])
        D = int(raw.get("n_demos", 0))

        hlog("=" * 60)
        hlog(f"Scenario           : {args.scenario}")
        hlog(f"Data file          : {data_path}")
        hlog(f"Units (N)          : {N}")
        hlog(f"Parameters (P)     : {P}")
        hlog(f"Demographics (D)   : {D}")
        hlog(f"K in data (K_TRUE) : {K_TRUE}")
        if raw.get("TRUE_PVEC") is not None:
            hlog(f"TRUE_PVEC          : {np.array(raw['TRUE_PVEC']).round(4)}")
        hlog("-" * 60)
        hlog(f"Sampler            : bayesm (rhierMnlRwMixture)")
        hlog(f"K_MODEL            : {args.k_model}")
        hlog(f"dirichlet_a        : {args.dirichlet_a}  | a_mu: {args.a_mu} | A_delta: {args.a_delta}")
        hlog(f"chains/R/thin/burn : {args.chains} / {args.r_total} / {args.thin} / {args.burn_in}")
        hlog(f"Rscript            : {RSCRIPT}")
        hlog("=" * 60)

        # ── 1. run the R sampler ───────────────────────────────────────────
        raw_dir.mkdir(parents=True, exist_ok=True)
        r_cmd = [
            RSCRIPT, str(R_SCRIPT),
            "--data-path",   str(data_path),
            "--out-raw-dir", str(raw_dir),
            "--scenario",    args.scenario,
            "--k-model",     str(args.k_model),
            "--chains",      str(args.chains),
            "--r-total",     str(args.r_total),
            "--burn-in",     str(args.burn_in),
            "--thin",        str(args.thin),
            "--seed",        str(args.seed),
            "--a-mu",        str(args.a_mu),
            "--a-delta",     str(args.a_delta),
            "--dirichlet-a", str(args.dirichlet_a),
            "--nprint",      str(args.nprint),
        ]
        hlog(f"Launching R sampler ...")
        # Stream R's stdout/stderr into sampling.log (the clean engine log).
        with open(sampling_log, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                r_cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),   # so the project .Rprofile activates renv
            )
        if proc.returncode != 0:
            tail = sampling_log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            raise RuntimeError(
                "R sampler failed (exit "
                f"{proc.returncode}). Last lines of sampling.log:\n" + "\n".join(tail)
            )

        # ── 2. load the raw draws ──────────────────────────────────────────
        with open(raw_dir / "dims.json") as f:
            dims = json.load(f)
        with open(raw_dir / "meta_r.json") as f:
            meta_r = json.load(f)
        # jsonlite auto-unboxes length-1 vectors to scalars, so a chains==1 run
        # yields a bare int where the per-chain list is expected. Force list form.
        def _as_list(x):
            if x is None:
                return []
            return list(x) if isinstance(x, (list, tuple)) else [x]
        for _k in ("seeds", "durations_s", "loglike_mean"):
            meta_r[_k] = _as_list(meta_r.get(_k))

        C   = int(dims["chains"])
        S   = int(dims["n_samples"])
        K   = int(dims["K"])
        pc  = dims["per_chain"]
        has_Z = bool(dims["has_Z"])

        def stack(name):
            shape = pc[name]
            return np.stack(
                [_read_bin(raw_dir / f"{name}_chain{c}.bin", shape) for c in range(C)],
                axis=0,
            )

        mu_k   = stack("mu")                     # (C, S, K, P)
        prec   = stack("prec")                   # (C, S, K, P, P)
        pvec   = stack("pvec")                   # (C, S, K)
        beta_i = stack("beta")                   # (C, S, N, P)
        hlog(f"Loaded raw draws: chains={C}  draws/chain={S}  K={K}  P={P}")

        # ── 3. rebuild the Liesel-format latent + assemble canonical dict ──
        sigma_inv_chol_k_latent = _precision_to_latent(prec)   # (C, S, K, P(P+1)/2)
        if not np.all(np.isfinite(sigma_inv_chol_k_latent)):
            raise FloatingPointError(
                "Non-finite values in reconstructed sigma_inv_chol_k_latent "
                "(precision -> Cholesky -> FillScaleTriL inverse) - likely a "
                "degenerate/empty mixture component in the bayesm draws."
            )

        canonical = {
            "mu_k": mu_k,
            "sigma_inv_chol_k_latent": sigma_inv_chol_k_latent,
            "pvec": pvec,
            "beta_i": beta_i,
        }
        if has_Z:
            canonical["Delta"] = stack("delta")  # (C, S, D, P)

        with open(outdir / "posterior_raw.pkl", "wb") as f:
            pickle.dump({k: np.asarray(v) for k, v in canonical.items()}, f)
        hlog("Saved posterior_raw.pkl")

        # ── 4. export.pkl via the SHARED exporter (mu/sigma/std/pvec) ──────
        try:
            from src.analysis import export_posterior_to_pickle
            export_posterior_to_pickle(
                samples=canonical, K=args.k_model, P=P,
                filename="export.pkl", output_dir=str(outdir),
            )
        except Exception as exp_err:
            hlog(f"WARNING export failed: {exp_err}")

        duration = time.time() - t0

        # ── 5. meta.json - same keys the notebook reads (warmup<-burn_in,
        #        posterior<-kept draws), plus bayesm-specific fields ─────────
        meta = {
            "scenario": args.scenario, "data_path": str(data_path),
            "k_model": args.k_model, "k_true": K_TRUE,
            "n_params": P, "n_units": N, "n_demos": D,
            "sampler": "bayesm", "chains": args.chains,
            "warmup": args.burn_in, "posterior": S,        # mapped for notebook display
            "seed": args.seed,
            "a_delta": args.a_delta, "a_mu": args.a_mu, "dirichlet_a": args.dirichlet_a,
            "r_total": args.r_total, "thin": args.thin, "burn_in": args.burn_in,
            "n_samples": S, "seeds": meta_r.get("seeds"),
            "bayesm_version": meta_r.get("bayesm_version"),
            "r_version": meta_r.get("r_version"),
            "chain_durations_s": meta_r.get("durations_s"),
            "loglike_mean": meta_r.get("loglike_mean"),
            "started_at": started_at, "duration_s": round(duration, 1),
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # ── 6. summary headline + per-chain timings/loglike ────────────────
        hlog("-" * 60)
        hlog("Per-chain summary (from R):")
        seeds = meta_r.get("seeds") or []
        durs  = meta_r.get("durations_s") or []
        lls   = meta_r.get("loglike_mean") or []
        for c in range(C):
            sd = seeds[c] if c < len(seeds) else "?"
            du = durs[c] if c < len(durs) else "?"
            ll = lls[c] if c < len(lls) else "?"
            hlog(f"   chain {c}  seed={sd:<6}  {du:>7}s  mean loglike={ll}")
        hlog("-" * 60)
        hlog(f"DONE in {datetime.timedelta(seconds=int(duration))}")
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        # ── 7. tidy up the big raw bins (redundant with posterior_raw.pkl) ─
        if not args.keep_raw:
            for b in raw_dir.glob("*.bin"):
                b.unlink()

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


if __name__ == "__main__":
    sys.exit(main())
