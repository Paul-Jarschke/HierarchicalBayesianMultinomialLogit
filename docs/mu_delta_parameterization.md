# Why `beta_i = mu + Z @ Delta + u_i`, not `beta_i = Z @ Delta + u_i` with `u_i` zero-mean

This note resolves an apparent conflict between Rossi, Allenby & McCulloch (2006)
§5.4's textbook formula and the parameterization actually used in this repo
(`src/standardmodel.py`, `src/mixturemodel.py`, `src/dgp.py`, and their `bayesm_*`
counterparts) and in bayesm's own software. It applies identically to the
single-component model (`mu`) and to each component of the mixture model (`mu_k`).

## The apparent tension

Rossi (2006) §5.4, equation (5.4.1), writes the hierarchical logit model as:

```
B = Z*Delta + U,                    u_i ~ N(0, V_beta)
vec(Delta | V_beta) ~ N(vec(Delta_bar), V_beta (x) A^-1)
V_beta ~ IW(nu, V)
```

Read literally, `U` has **zero mean**, so `Z*Delta` alone must be the entire
conditional mean of `beta_i`. There is no separate `mu` symbol anywhere in
(5.4.1).

This repo's model (and bayesm's actual `rhierMnlRwMixture` implementation)
instead has:

```
beta_i = Z[i] @ Delta + u_i,     u_i ~ N(mu, Sigma)     # non-zero mean
```

i.e. a separate `mu`, added on top of `Z @ Delta`, with its own (Sigma-scaled)
prior. These look like different models. They are not — the difference is
entirely in whether `Z` carries an intercept column.

## What bayesm's software actually implements

Verified directly against the installed package (`?rhierMnlRwMixture` and
`print(rhierMnlRwMixture)`), not just the book:

```
beta_i = Z[i,] %*% Delta + u_i
u_i ~ N(mu_ind, Sigma_ind),   ind ~ Multinomial(pvec)

delta = vec(Delta) ~ N(deltabar, Ad^-1)          # fixed covariance, NOT Sigma-scaled
mu_j  ~ N(mubar, Sigma_j (x) Amu^-1)              # Sigma-scaled
Sigma_j ~ IW(nu, V)

Note: Z should NOT include an intercept and is centered for ease of interpretation.
```

Reading the R source confirms this prior structure (separate `Amu`/`Ad`, both
defaulting to `0.01`, `nu = nvar + 3`, `V = nu*I`) is built identically
regardless of `Prior$ncomp` — nothing folds `mu` into `Delta` when `ncomp = 1`.
The package's own bundled example simulates data the same way:

```r
Z = t(t(Z) - apply(Z, 2, mean))            # demeaned, no intercept column
Delta = matrix(c(1, 0, 1, 0, 1, 2), ncol = 2)   # nz x nvar - Z-coefficients only
comps[[1]] = list(mu = c(0, -1, -2), rooti = diag(rep(1, 3)))  # mu kept separate
betai = Delta %*% Z[i,] + as.vector(rmixture(1, pvec, comps)$x)
```

## Reconciling the two

Both describe the same model; they just draw the line between "deterministic
mean" and "zero-mean noise" in different places, depending on whether Z carries
an intercept.

Let `Z_tilde = [1, z_i]` (intercept prepended) and let `Delta_tilde` be the
corresponding `(1 + n_demos) x n_params` coefficient matrix, whose first row is
the population mean. Rename that first row `mu` and the remaining rows `Delta`
(now `n_demos x n_params`, matching Z **without** the intercept):

```
Delta_tilde' z_tilde_i = Delta_tilde[0]*1 + Delta_tilde[1:]' z_i
                        = mu + Delta' z_i

beta_i = Delta_tilde' z_tilde_i + U,      U ~ N(0, V_beta)        # (5.4.1), Z has intercept
       = mu + Z @ Delta + U,              U ~ N(0, Sigma)          # substitute
       = Z @ Delta + u_i,                 u_i := mu + U ~ N(mu, Sigma)   # bayesm's parameterization
```

Same distribution for `beta_i` throughout. (5.4.1)'s reading is correct *if* Z
includes an intercept and Delta's first row absorbs the population mean. Once Z
excludes the intercept — bayesm's explicit, documented requirement ("Z should
NOT include an intercept"), and this repo's convention throughout
(`src/dgp.py`, `src/mixturemodel.py`, `src/standardmodel.py`) — that role can't
disappear. It has to live somewhere, and bayesm puts it in `mu`: a separate
parameter with its own Sigma-scaled prior, added back on top of `Z @ Delta`.

The two-constants prior split (`mu` Sigma-scaled by `a_mu`/`Amu`; `Delta`
fixed-scale by `A_delta`/`Ad`) is not incidental to this — it's *why* bayesm
needs `mu` as a separate symbol at all. In the mixture generalization (§5.5),
each component needs its own location `mu_k`, but `Delta` (the demographic
effect) is shared across all components. That forces the intercept role out of
a single `Delta` and into per-component `mu_k`'s. The same `rhierMnlRwMixture`
function (and prior-construction code) is used for `ncomp = 1`, so the
single-component model inherits this split unchanged.

## Conclusion for this codebase

`beta_i = mu + Z @ Delta + u_i` (`u_i ~ N(0, Sigma)`) in
[`src/standardmodel.py`](../src/standardmodel.py) and
`beta_i = mu_k + Z @ Delta + u_i` in [`src/mixturemodel.py`](../src/mixturemodel.py)
are the correct translations of (5.4.1)/(5.5.x) into bayesm's actual,
no-intercept-Z parameterization — matching the installed package's real
behavior, not just its book description. `Z @ Delta` alone is *not* "the
mean" under this convention; `mu + Z @ Delta` is. This is why `Z` is stored
column-centred with no intercept column throughout `src/dgp.py`, and why `mu`
(or `mu_k`) and `Delta` carry different prior treatments (`a_mu`-scaled by
`Sigma`; `A_delta`-fixed) rather than one shared prior.
