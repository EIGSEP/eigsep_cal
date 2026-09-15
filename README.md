# eigsep_cal

EIGSEP instrument model and staged Bayesian calibration.

The package is being rebuilt. It will hold:

- an instrument forward model from antenna temperature to measured power
  per switch state (noise waves coupled through the reflection
  coefficients, noise source, gain drift, switch-path S-parameters,
  additive post-switch terms);
- stage solvers that return posteriors, not point estimates, so later
  stages can carry the uncertainty forward;
- nothing that reads data files or models the sky. Data loaders live in
  `data-analysis`; sky, beam and horizon simulation lives in
  `eigsep_mock_analysis` (eigsim), which imports this package to generate
  synthetic data.

Today it contains `dicke.calc_Tant_star` (Monsalve et al. 2017, eq. 1).

The 2024 Vivaldi gain calibration (`vivaldi_cal/`) and the `eigsep_corr`
dependency were removed; they are preserved at tag `legacy-2024`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```
