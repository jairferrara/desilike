# desilike.profilers

Likelihood profiling and parameter inference for cosmological analyses. This module provides four optimization backends that share a common interface for maximizing likelihoods, computing confidence intervals, generating 1D profile likelihoods, 2D contours, and grid scans.

## Available profilers

| Class | Backend | Derivative-free | Notes |
|---|---|---|---|
| `MinuitProfiler` | [iMinuit](https://github.com/scikit-hep/iminuit) | Yes (MIGRAD) | HEP standard; native MINOS intervals |
| `ScipyProfiler` | [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) | Optional | General purpose; many method options |
| `BOBYQAProfiler` | [pybobyqa](https://github.com/numericalalgorithmsgroup/pybobyqa) | Yes | Quadratic approximation; good for noisy likelihoods |
| `OptaxProfiler` | [optax](https://github.com/google-deepmind/optax) | No (requires JAX) | Gradient-based; best for large-scale JAX models |

## Quick start

```python
from desilike.profilers import MinuitProfiler
from desilike.likelihoods import ObservablesGaussianLikelihood

# Build your likelihood as usual
likelihood = ObservablesGaussianLikelihood(observables=[...])

# Create a profiler and find the best fit
profiler = MinuitProfiler(likelihood, seed=42)
profiles = profiler.maximize(niterations=4)

print(profiles.bestfit)           # best-fit parameter values
print(profiles.bestfit.logposterior)  # log-posterior at best fit

# 1D profile likelihood
profiler.profile(params=['omega_cdm'], size=30)

# 2D contour at 2-sigma
profiler.contour(params=[['omega_cdm', 'h']], cl=2)

# Confidence interval for a single parameter
profiler.interval(params=['omega_cdm'], cl=1)
```

All results accumulate in `profiler.profiles` and are returned by each method.

## Profiler constructor

All four profilers share the same base constructor:

```python
profiler = MinuitProfiler(
    likelihood,            # BaseLikelihood — required
    rng=None,              # numpy RNG or None
    seed=None,             # integer seed for reproducibility
    max_tries=1000,        # max attempts to find a valid starting point
    profiles=None,         # existing Profiles to append to
    ref_scale=1.,          # scale factor applied to ref/proposal widths
    rescale=False,         # rescale parameters internally to unit variance
    covariance=None,       # external covariance for rescaling / error estimates
    save_fn=None,          # path — save profiles after each operation
    mpicomm=None,          # MPI communicator for parallel runs
)
```

`MinuitProfiler` and `ScipyProfiler` additionally accept `gradient=False` to enable gradient-assisted minimization when a JAX-compatible likelihood is provided. `ScipyProfiler` also accepts `method=None` to pick a specific scipy optimizer (e.g. `'L-BFGS-B'`, `'Nelder-Mead'`). `OptaxProfiler` accepts `method='adam'` to select the optax optimizer.

## Core methods

### `maximize`

```python
profiles = profiler.maximize(niterations=1, start=None, **kwargs)
```

Runs the optimizer `niterations` times from different starting points drawn from the parameter reference distribution and returns the result with the lowest chi-squared. Stored in `profiles`: `start`, `bestfit`, `error`, `covariance`.

The `start` argument can be a dict of parameter values to use as a fixed starting point instead of sampling.

Keyword arguments are forwarded to the backend (e.g. `max_iterations=100000`).

### `profile`

```python
profiles = profiler.profile(params=None, grid=None, size=30, cl=2, niterations=1, **kwargs)
```

Computes 1D profile likelihoods. For each parameter value on the grid, all other varied parameters are re-optimized. Stored in `profiles`: `profile`.

- `params`: list of parameter names; defaults to all varied parameters.
- `grid`: dict mapping parameter name → array of values; if omitted, a grid is auto-generated.
- `size`: number of grid points per parameter.
- `cl`: confidence level (in sigma) used to set the grid range.
- `niterations`: re-optimization restarts per grid point.

### `interval`

```python
profiles = profiler.interval(params=None, cl=1, niterations=1, **kwargs)
```

Computes frequentist confidence intervals by finding where the profile chi-squared increases by `delta_chi2 = cl^2` (for a 1D interval). Stored in `profiles`: `interval`.

`cl` can be a probability in (0, 1) or a number of sigma ≥ 1.

### `contour`

```python
profiles = profiler.contour(params=None, cl=1, niterations=1, size=50, **kwargs)
```

Computes 2D confidence contours. `params` is a list of 2-element lists, e.g. `[['omega_cdm', 'h'], ['omega_cdm', 'omega_b']]`. If a single flat list of parameter names is given, all unique pairs are computed. Stored in `profiles`: `contour`.

### `grid`

```python
profiles = profiler.grid(params=None, grid=None, size=1, cl=2, niterations=1, **kwargs)
```

Evaluates the full likelihood (with re-optimization of all other parameters) on a parameter grid. Stored in `profiles`: `grid`.

### `covariance`

```python
profiles = profiler.covariance(**kwargs)
```

Estimates the parameter covariance from the Fisher information matrix at the best-fit point. Requires `maximize` to have been run first. Stored in `profiles`: `error`, `covariance`.

## The `Profiles` object

Every method returns (and appends to) a `Profiles` object. Key attributes:

| Attribute | Description |
|---|---|
| `bestfit` | `ParameterBestFit` — best-fit values for all parameters plus `logposterior` |
| `error` | Standard errors at the best fit |
| `covariance` | Full parameter covariance matrix |
| `start` | `Samples` — the starting points tried per iteration |
| `interval` | Confidence intervals (lower, upper) per parameter |
| `profile` | 1D profile likelihood curves per parameter |
| `contour` | 2D contour points per parameter pair |
| `grid` | Grid of logposterior values |

Profiles can be saved and loaded:

```python
profiler = MinuitProfiler(likelihood, save_fn='profiles.npy')
# ...
from desilike.samples import Profiles
profiles = Profiles.load('profiles.npy')
```

## Parameter rescaling

Setting `rescale=True` transforms parameters internally to unit variance before passing them to the optimizer, then maps results back. This can improve convergence when parameters have very different scales:

```python
profiler = MinuitProfiler(likelihood, rescale=True, covariance='path/to/covariance.npy')
```

If no covariance is supplied, the parameter proposal widths are used instead.

## MPI parallelization

Pass an MPI communicator to distribute independent optimization restarts across ranks:

```python
from mpi4py import MPI
profiler = MinuitProfiler(likelihood, mpicomm=MPI.COMM_WORLD)
profiles = profiler.maximize(niterations=8)   # 8 starts distributed across ranks
```

Grid and contour evaluations are also parallelized: grid points are distributed across ranks, with each rank performing its own re-optimization.

## Choosing a profiler

- **`MinuitProfiler`** is the default recommendation. MIGRAD is reliable, the HESSE step gives a covariance estimate, and MINOS provides profile-likelihood intervals without additional setup.
- **`ScipyProfiler`** is useful when you need a specific method not available in Minuit (e.g. `'trust-constr'` for constrained optimization) or when Minuit is unavailable.
- **`BOBYQAProfiler`** is well-suited to likelihood evaluations that are noisy or non-differentiable (e.g. numerical integrals, emulators). It uses a derivative-free quadratic surrogate model.
- **`OptaxProfiler`** is designed for JAX-based likelihoods where automatic differentiation is available. It supports learning rate scheduling and early stopping.

## Dependencies

| Profiler | Required package |
|---|---|
| `MinuitProfiler` | `iminuit` |
| `ScipyProfiler` | `scipy` |
| `BOBYQAProfiler` | `pybobyqa` |
| `OptaxProfiler` | `optax`, `jax` |

The base module (`BaseProfiler`) optionally uses JAX for JIT compilation and automatic gradients; it falls back gracefully if JAX is not installed.
