"""Plot the three Liesel model graphs (Graphviz `dot` layout) into model_graphs/."""

import pathlib

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import liesel.model as lsl

from src.mixturemodel import build_mixture_hbmnl_model
from src.bayesm_mixture_model import build_bayesm_mixture_hbmnl_model
from src.standardmodel import build_standard_hbmnl_model

rng = np.random.default_rng(0)
DATA = {
    "X":        jnp.array(rng.normal(size=(12, 3, 3))),
    "y":        jnp.array(rng.integers(0, 3, 12)),
    "unit_idx": jnp.array(np.repeat(np.arange(6), 2)),
    "Z":        jnp.array(rng.normal(size=(6, 2))),
    "n_params": 3,
    "n_units":  6,
}

MODELS = {
    "mixture":   build_mixture_hbmnl_model(DATA, K=3),        # NUTS / HMC (marginalized)
    "augmented": build_bayesm_mixture_hbmnl_model(DATA, K=3),  # replication (explicit ind)
    "standard":  build_standard_hbmnl_model(DATA),             # one-component
}

outdir = pathlib.Path("model_graphs")
outdir.mkdir(exist_ok=True)

for name, model in MODELS.items():
    lsl.plot_vars(model, show=False, width=17, height=12, prog="dot")
    plt.gcf().savefig(outdir / f"model_graph_{name}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"wrote model_graphs/model_graph_{name}.png")
