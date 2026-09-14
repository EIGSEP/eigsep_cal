# eigsep_cal API spec v0

*Status: draft, 2026-09-14. Owner: Christian Hellum Bye.*

This spec fixes eigsep_cal's conventions, forward model, data objects and stage API, so that code producing its inputs (data adapters, synthetic-data generators) and code consuming its outputs can be written against it.

Items marked **Open** need a decision before code depends on them. Once a consumer is built on this spec, every change bumps the version and gets a line in the changelog (§ 12).

**Background**
- Calibration is staged and Bayesian. Each stage passes a posterior, not a point estimate; there is no single joint fit.
- The physics is derived and checked in [memo M003](memos/M003_receiver_calibration/memo.md). Every equation used here is written out in § 4 under the memo's names (eq. Ps, eq. map, …).
- **Section numbers** match the Deployment 5 interface spec, which builds on this one and covers eigsim, the synthetic-data generator and the D5 adapters. Its sections 7–9 have no counterpart here.

---

## 1. Scope

- **eigsep_cal** depends on numpy. The v0 stage solvers are closed-form Gaussian: the drift model of 3a is linear in its basis coefficients, and given Q_s the calibration equation of 3b is linear in Θ. SciPy, JAX or a sampler come in only when a nonlinear block needs them.
- It never reads instrument data files, and never models sky, beam or terrain. It knows nothing about the quirks of any one deployment. It may save and load its own objects (§ 5.6).
- **Inputs** come from a data adapter or a synthetic-data generator. Both emit the same objects (§ 5), so a stage cannot tell synthetic input from real input.

---

## 2. Conventions

| Quantity | Convention |
|---|---|
| Frequency | `freqs_mhz`: float64, MHz, strictly increasing |
| Time | `times_unix`: float64, seconds since 1970-01-01 UTC |
| Temperature | Kelvin, float64 |
| Power | Linear, arbitrary units (for example correlator counts), float64. Never dB. |
| Reflection coefficient | complex128, 50 Ω reference, looking into the device from the reference plane P (§ 3) |
| Array axes | Leading batch axes, then time, then frequency: `(..., n_time, n_freq)` |
| Flags | bool; `True` means bad (hera convention) |
| Precision | float64 and complex128 throughout |

**Frequency grids**
- **No resampling inside eigsep_cal.** All arrays passed to one call share an identical `freqs_mhz` array, checked with `np.array_equal`. eigsep_cal never interpolates in frequency.
- **Resampling happens upstream.** For example, an adapter maps the VNA's S11 grid onto correlator channels, and the input's uncertainty carries the interpolation error.
- **Correlator channel grid:** `f_k = k * 250 / 1024` MHz for k = 0…1023, so Δν = 0.244140625 MHz (`eigsep_observing.utils.calc_freqs_dfreq`).
- **Synthetic studies use a subset of these channels** and carry the integer channel indices (`chan`), so results compare with data channel by channel.

---

## 3. Reference plane and switch states

**Reference plane P** is the common LNA-side node of the switch network.
- Every state's switch path is part of its source: Γ_s by embedding, T_s by eq. availgain (§ 4.3).
- The receiver reflection Γ_rec is measured at P.

**State names** follow `eigsep_observing.client.OBS_MODES`; the rfswitch paths are `picohost.base.PicoRFSwitch.PATHS`.

| State | Physical source behind its path | T_s at P from | VNA key | VNA path |
|---|---|---|---|---|
| `RFANT` | antenna, balun and balun–switch coax, as one source (see below) | `SkyTemperature` through balun and coax (generator only, § 4.3), then the RFANT path | `ant` | VNAANT |
| `RFAMB` | 50 Ω ambient load | load temperature | `amb` | VNAAMB |
| `RFNON` | noise diode on, behind the pad | diode excess plus pad (§ 4.3) | `noise` | VNANON |
| `RFNOFF` | noise diode off, behind the pad | pad/diode physical temperature | `load` | VNANOFF |
| `RFSP1_SHORT`, `RFSP1_OPEN` | long coax ending in a short or open | cable temperature | `sp1_short`, `sp1_open` | VNASP1 |
| receiver | LNA input | — | `rec` (in `recs11`) | VNARF |

**The antenna includes the balun and the balun–switch coax**
- The switch comes after the coax from the antenna balun. The S11 chain and the in-situ calibration de-embed only switch paths, so for both of them the coax is part of the antenna. The measured Γ_ant includes the balun and coax, and so does the calibrated antenna temperature (§ 6).
- Free-space beam models (HFSS, eigsim) include no balun loss and no coax. A simulated `t_ant_k` (§ 5.1) therefore sits at a different plane from anything eigsep_cal measures or calibrates.
- Calibration does not need the balun or coax S-parameters. Comparisons with simulations do: the generator (§ 4.3) and stage 5 model both, with priors where there are no measurements.

**Traps and scope**
- The VNA key `load` is the noise diode **off** (`cmt_vna.VNA.measure_ant` switches to VNANOFF). The ambient load is `amb`.
- Rows labelled `VNA*`, `UNKNOWN` or `MISSING` never reach a stage. Whoever builds the `Observation` either resolves them to a state from the spectra or drops them, and records the count in `provenance`. v0 has no latent-state model (§ 11).

---

## 4. Physics in the forward model

### 4.1 Power per state
For a switch state s whose signal reaches the amplifier through P:

```
P_s(ν, t) = g_s(ν, t) · [ M_s T_s + |Γ_s F_s|² T_unc + Re(Γ_s F_s) T_cos
                           + Im(Γ_s F_s) T_sin + T_0 ] + A_s(ν, t)        (eq. Ps)

F_s = sqrt(1 − |Γ_rec|²) / (1 − Γ_s Γ_rec),    M_s = (1 − |Γ_s|²) |F_s|²    (eq. F)
```

- **Terms.**
  - g is the gain referenced to P, and T_s is the available noise temperature of the source at P.
  - M_s is the mismatch factor: delivered power over available power.
  - T_0 is receiver noise delivered independently of the source. T_unc, T_cos and T_sin describe receiver noise emitted toward the source and reflected back, in parts uncorrelated and correlated with T_0.
- **Provenance.** Without A_s this is Rogers & Bowman (2012) eq. 8 and Monsalve et al. (2017) eq. 2, since |Γ_s| |F_s| cos α = Re(Γ_s F_s) with α = arg(Γ_s F_s).
- **Assumptions.**
  - The receiver is linear.
  - g, T_0 and the noise parameters are stable over one switching cycle.
  - Source noise is uncorrelated with receiver noise.
  - Switch positions not in use are perfectly isolated.
  - T_s includes any lossy path between the physical source and P (eq. availgain, § 4.3).
- g_s = g · r_s, where r_s is a path-gain ratio. It is 1 by default, and the generator can perturb it. Only gain changes common to all paths cancel in Q_s (eq. Q, § 4.5), so the fits need stable ratios.
- A_s is zero in the v0 fit. The generator can switch it on, for example for a state-independent comb injected after the switch.

### 4.2 Receiver parameters: `ReceiverModel`

| Field | Shape | Notes |
|---|---|---|
| `t_unc_k`, `t_cos_k`, `t_sin_k`, `t0_k` | `(n_freq,)` or `(n_time, n_freq)` | Rogers & Bowman parameterisation |
| `t_r_k`, `t_l_k`, `c_k` | same; `c_k` complex | Intrinsic parameterisation (eq. map, below). Use either this set or the row above, selected by `parameterisation`. |
| `gamma_rec` | `(n_freq,)` or `(n_time, n_freq)` complex | Measured at sparse epochs; mapping onto samples is open (§ 11) |
| `gain` | `(n_time, n_freq)` | Must be > 0 |
| `path_gain_ratio` | `dict[state, (n_freq,)]` | Defaults to 1 |

In the generator, any of these may be a function of a named covariate from `Observation.covariates`, for example a temperature coefficient.

**Intrinsic parameterisation** (Bucher et al. 2026, arXiv:2607.26741, following Meys 1978). The amplifier is a noiseless two-port behind input-referred travelling-wave noise sources:
- A_R moves into the amplifier and A_L moves toward the source;
- T_R = ⟨|A_R|²⟩, T_L = ⟨|A_L|²⟩, and c = ⟨A_R* A_L⟩.

They map onto the Rogers & Bowman set as

```
T_0             = (1 − |Γ_rec|²) T_R
T_unc           = T_L + |Γ_rec|² T_R + 2 Re(Γ_rec c*)
T_cos − i T_sin = 2 sqrt(1 − |Γ_rec|²) (c + Γ_rec T_R)                     (eq. map)
```

- The coefficients depend on Γ_rec only, never on Γ_s, so eq. Ps with eq. map is the same model in other coordinates.
- The intrinsic set stays valid when Γ_rec changes through the amplifier's reverse transfer. The Rogers & Bowman set must then be re-mapped.
- Bucher et al. print T_unc without the factor 2 in the cross term. That version fails the § 10 test.

### 4.3 Source temperatures
- **Path embedding.** `embed(gamma_term, t_term_k, sparams, t_path_k) -> (gamma_s, t_s_k)` is for a termination (Γ_t, T_t) on port 2 of a path at physical temperature T_p (port convention § 5.2):

  ```
  Γ_s = S11 + S12·S21 Γ_t / (1 − S22 Γ_t)
  T_s = G T_t + (1 − G) T_p,
  G   = |S21|² (1 − |Γ_t|²) / ( |1 − S22 Γ_t|² (1 − |Γ_s|²) )               (eq. availgain)
  ```

  - G is the available gain of the path (Monsalve et al. 2017, eqs. 8–9).
  - |S21|² is taken as |S12·S21|, which assumes the path is reciprocal and passive.
- **Noise source** (generator only).
  - Diode excess temperature: T_ex = 290 K · 10^(ENR/10).
  - Behind a matched pad with loss factor L ≥ 1 at physical temperature T_pad, the source temperature before the path is (T_diode + T_ex) / L + (1 − 1/L) · T_pad.
- **Fits never need this.** They use the effective constants T_NS and T_L (eq. TNSTL, § 4.5), with a prior on T_NS.
- **Balun and coax** (generator only).
  - The generator turns a free-space `t_ant_k` into the RFANT source with `embed`: first through a lossy balun two-port, then through the coax two-port at its physical temperature, then through the RFANT switch path. `gamma_term` is the free-space antenna reflection.
  - Where the two-ports are not measured, their parameters are drawn from a prior (cable type and length, datasheet loss, balun loss, cable temperature), and the draw is recorded in `provenance`.
  - Fits never need them, because the measured Γ_ant already includes both.

### 4.4 Noise
- **Radiometer noise.** Per sample, σ² = P² / (B_eff · τ · n_int).
- **Bandwidth.** `enbw_hz` defaults to Δν = 244 140.625 Hz, until the SNAP polyphase filterbank's equivalent noise bandwidth is measured (§ 11).
- **Channel correlation.** Channels are independent in v0.

### 4.5 Calibration equation (stage 3)
Stage 3 removes the gain with two internal references, L = `RFAMB` (load) and N = `RFNON` (noise source on):

```
Q_s = (P_s − P_L) / (P_N − P_L)                                              (eq. Q)
```

Write P_L = g_L D_L and P_N = g_N D_N, where D is the bracket of eq. Ps for each reference on its own path. Dividing by g (1 − |Γ_rec|²) gives, for every source s:

```
T_s (1 − |Γ_s|²) / |1 − Γ_s Γ_rec|²  +  T_unc |Γ_s|² / |1 − Γ_s Γ_rec|²
  + T_cos Re[Γ_s / (1 − Γ_s Γ_rec)] / sqrt(1 − |Γ_rec|²)
  + T_sin Im[Γ_s / (1 − Γ_s Γ_rec)] / sqrt(1 − |Γ_rec|²)
  = Q_s T_NS + T_L                                                           (eq. cal)

T_NS = (g_N D_N − g_L D_L) / (g (1 − |Γ_rec|²)),
T_L  = ((g_L / g) D_L − T_0) / (1 − |Γ_rec|²)                                (eq. TNSTL)
```

- **Exact and linear.** Eq. cal is exact under the assumptions of § 4.1, and linear in Θ = (T_unc, T_cos, T_sin, T_NS, T_L).
- **What the references need.** They need not be matched, nor sit on the same switch port as the calibrators. Only gain changes common to all paths cancel, so g_L/g and g_N/g must be stable.
- **Drift.** T_NS and T_L contain the references' physical temperatures through D_L and D_N. Without temperature control, model them as linear in the logged temperatures, for example T_L(t) = T_L⁰ + κ_L [T_amb(t) − mean(T_amb)]. The system stays linear.
- **T_NS is more than the diode ENR.** It also holds the pad attenuation, the mismatch between N and L, g_N/g_L, and, when L and N are on separate ports, the pad temperature minus the load temperature.
- **Monsalve et al. (2017) C_1 and C_2** are the same two constants: T_NS = C_1 T_NS^a and T_L = T_L^a − C_2.

**Design-matrix form** (Roque et al. 2021). Multiplying eq. cal by |1 − Γ_s Γ_rec|² / (1 − |Γ_s|²) gives T_s = X_L T_L + X_NS T_NS + X_unc T_unc + X_cos T_cos + X_sin T_sin, with

```
X_L   =  |1 − Γ_s Γ_rec|² / (1 − |Γ_s|²)
X_NS  =  Q_s X_L
X_unc = −|Γ_s|² / (1 − |Γ_s|²)
X_cos = −Re[Γ_s (1 − Γ_s* Γ_rec*)] / ((1 − |Γ_s|²) sqrt(1 − |Γ_rec|²))
X_sin = −Im[Γ_s (1 − Γ_s* Γ_rec*)] / ((1 − |Γ_s|²) sqrt(1 − |Γ_rec|²))       (eq. X)
```

A calibrator with known T_s contributes a row. The antenna temperature follows from the same expression, T_ant = X_ant Θ̂.

---

## 5. Objects

These are dataclasses of numpy arrays, validated on construction (shapes, dtypes, identical frequency grids) and treated as immutable. Names are provisional until track A's first commit.

### 5.1 `SkyTemperature`
Built by the generator from sky-simulation output; consumed by the forward model.

| Field | Shape | Notes |
|---|---|---|
| `t_ant_k` | `(n_time, n_freq)` | Available noise temperature at the terminals of the free-space antenna model: everything the simulator models, meaning sky, horizon and ground. **No receiver term, no balun loss, no coax.** The generator adds the balun and everything after it with `embed` (§ 4.3). This is not the plane of a calibrated spectrum (§ 3). |
| `freqs_mhz` | `(n_freq,)` | |
| `times_unix` | `(n_time,)` | |
| `antenna` | str | `"box-air"` or `"box-gnd"` |
| `elevation_deg`, `azimuth_deg` | `(n_time,)` | Drive angles per sample; provenance only |
| `meta` | dict | Simulator version and config; beam, horizon, sky and ground models, with any drawn parameters |

### 5.2 `SParams`

| Field | Shape | Notes |
|---|---|---|
| `s11`, `s12s21`, `s22` | `(n_freq,)` complex | Same layout as the rows of `switch_sparams.npz`: `[S11, S12·S21, S22]` |
| `name` | str | For example `"RFANT"` or `"VNAANT"` |

**Port convention:** cmt_vna `calkit`'s, which data-analysis's S11 chain (`scripts/calibrate_field_s11.py`) uses.
- Γ' = S11 + S12·S21 Γ / (1 − S22 Γ): port 1 (`s11`) faces the reference side, and port 2 (`s22`) faces the termination.
- The reference side is P for `RF*` paths and the VNA for `VNA*` paths.

### 5.3 `Reflection`
The output of stage 2 (S11 calibration, which lives outside eigsep_cal, § 6), consumed by stage 3.

| Field | Shape | Notes |
|---|---|---|
| `gamma` | `(n_meas, n_freq)` complex | At P. For `RFANT` it includes the balun and coax (§ 3). |
| `gamma_cov` | `(n_meas, n_freq, 2, 2)` | Covariance of (Re, Im) |
| `times_unix` | `(n_meas,)` | Measurement time: S11 `metadata_snapshot_unix`, not the filename |
| `source` | str | A § 3 state name, or `"receiver"` |
| `freqs_mhz` | `(n_freq,)` | |

### 5.4 `Observation`
The data vector for one antenna, from an adapter or the generator.

| Field | Shape | Notes |
|---|---|---|
| `power` | `(n_time, n_freq)` | One integration or an average of several |
| `state` | `(n_time,)` str | § 3 names only |
| `times_unix` | `(n_time,)` | Sample centre |
| `tau_s` | `(n_time,)` | Time per integration |
| `n_int` | `(n_time,)` int | Integrations averaged into the sample |
| `flags` | `(n_time, n_freq)` bool | `True` = bad |
| `freqs_mhz` | `(n_freq,)` | |
| `chan` | `(n_freq,)` int or `None` | Correlator channel index |
| `antenna` | str | |
| `covariates` | `dict[str, (n_time,)]` | Physical temperatures in K, NaN where missing, for example `t_amb_k` and `t_switch_k` |
| `changepoints_unix` | `(n_cp,)` | Epochs where drift parameters may jump, taken from the deployment timeline |
| `provenance` | dict | Input files and selection, or the generator's truth config |

Averages must not cross a state change or a changepoint; split the sample there instead.

### 5.5 `Posterior`
The object carried between stages. Parameters are grouped into named **blocks**, and each block is a function of (frequency, time) expanded in a declared basis.

| Field | Content |
|---|---|
| `blocks` | A list of `Block(name, basis, slice, units)`. `basis` declares, for example, a Chebyshev order over [f_lo, f_hi] and piecewise-linear time dependence between changepoints. |
| `mean` | `(n_par,)` |
| `cov` | `(n_par, n_par)` or `None` (Gaussian form) |
| `samples` | `(n_samp, n_par)` or `None` (sampled form). At least one of `cov` and `samples` is set. |
| `prior` | The same form as the posterior, plus a `source` per block: datasheet, lab, or notebook ID |
| `diagnostics` | Per block: the prior-to-posterior variance ratio (minimum and median over coefficients); KL(posterior ‖ prior) in the Gaussian case; condition number; χ²/dof; the held-out checks that were run |
| `stage`, `inputs` | Stage ID, and the provenance of every input, including upstream posteriors by content hash |

**Required methods**
- `evaluate(name, freqs_mhz, times_unix) -> (mean, cov)` returns a block on any grid. `cov` is over the flattened `(n_time · n_freq)` grid.
- `draw(name, freqs_mhz, times_unix, n, rng) -> (n, n_time, n_freq)` returns samples on a grid.
- **Consumers use only these two methods.** They never read coefficients directly, so a stage can change its basis without breaking anything downstream.

### 5.6 Persistence
`Posterior`, `Reflection`, `Observation` and `CalibratedSpectrum` provide `save(path)` and `load(path)` to an npz or HDF5 file that includes a `spec_version` field. This is eigsep_cal's only file I/O.

---

## 6. Stage API (eigsep_cal, v0)

```python
# Forward model: the generator builds data with it; fits use it to predict.
embed(gamma_term, t_term_k, sparams, t_path_k) -> (gamma_s, t_s_k)
power(sources, receiver, state, times_unix) -> power           # noiseless
radiometer_noise(power, enbw_hz, tau_s, n_int, rng) -> noise

# Stage 3a: the reference drift model.
fit_references(obs, prior) -> Posterior
# Fits P_AMB and P_NON as functions of (ν, t), with temperature covariates
# and changepoints, so the switched ratio Q_s (eq. Q, § 4.5) and its
# uncertainty are available at every non-reference sample.

# Stage 3b: noise waves and reference constants.
fit_noise_waves(obs, reflections, post_3a, prior) -> Posterior
# Fits Θ = (T_unc, T_cos, T_sin, T_NS, T_L) plus drift coefficients
# from the calibrator states (eqs. cal and X, § 4.5), with an explicit
# T_NS prior.

calibrate(obs, reflections, post_3a, post_3b) -> CalibratedSpectrum
predict(post_3a, post_3b, state, reflection, covariates, times_unix)
    -> (mean, cov)   # predictive power for held-out checks
```

- `sources` is a `dict[state, Source]`. A `Source` holds `gamma_s` and `t_s_k`, each shaped `(1 or n_time, n_freq)`.
- `CalibratedSpectrum` holds:
  - `t_ant_k` mean, `(n_time, n_freq)`;
  - either `cov` over frequency per time sample, or `draws`;
  - `flags`, `antenna`, `freqs_mhz`, `times_unix`, `provenance`.

  It is the input to stage 5.
- **`CalibratedSpectrum.t_ant_k` is not a simulated free-space `t_ant_k`.** It is the RFANT source temperature at P, covering the antenna, balun, coax and RFANT switch path (§ 3). Measured S-parameters can remove the switch path; nothing in the calibration removes the balun and coax. To compare with simulations, stage 5 forward-models them and marginalises over their priors (§ 4.3).
- 3a comes before 3b because the model is bilinear in gain and noise waves. The v0 validation (§ 10) must compare the two-stage result with a joint fit on synthetic data, to measure what the split costs.

**Stage 2 (S11 calibration) lives in data-analysis, not in eigsep_cal.**
- `scripts/calibrate_field_s11.py` already runs the chain with cmt_vna `calkit`: raw → internal open/short/load (`vna` plane) → de-embed the `VNA*` path (`dut`) → embed the `RF*` path (`lna`, which is P).
- Keeping one implementation avoids two versions of the same calibration drifting apart.
- The chain gives point estimates. The adapter wraps the `lna` plane as `Reflection` and supplies `gamma_cov`, for example from the scatter between repeat captures, or from Monte Carlo through the same chain with perturbed inputs (standard models, switch-path S-parameters, resampling onto correlator channels).
- eigsep_cal's `embed` repeats the Γ cascade, because the forward model also needs T_s. The generator package tests it against `calkit.embed_sparams`, so the two cannot drift apart.

---

## 10. Validation contract (every stage)

- **Coverage.**
  - Run ≥ 200 synthetic realisations at the deployment's switching cadence, with its gaps and calibrator epochs.
  - Per block, the fraction of truths inside the central 68 % and 95 % intervals must agree with nominal at the 99 % binomial level.
  - Report it once with the generator's effects beyond the fit model off and once with them on.
- **Prior against posterior.** Report the § 5.5 diagnostics for every block.
- **Held-out prediction.** Score `predict` against data kept out of the fit (for example one SP1 termination, or next-night AMB/NON) as χ² against the predictive covariance.
- **Forward-model tests in CI.**
  - A unit test checks eq. Ps, with eq. map, against a direct solution of the four wave equations for random networks, to 1e-11 K.
  - The map without the factor 2 in T_unc must fail that test.

---

## 11. Open

| Item | Needed by |
|---|---|
| SNAP channel equivalent noise bandwidth and neighbour correlation | Noise model (§ 4.4) |
| Latent switch state for `MISSING` rows | v1 |
| Structured or sparse covariance for large blocks (dense above ~10⁴ parameters is too big) | v1 |
| Time-varying path-gain ratios. § 4.2 and `ReceiverModel` accept `path_gain_ratio` as `(n_freq,)` only, but a generator needs drifting r_s to test the fit. Allow `(n_time, n_freq)`. Folding r_s(t) into `gain` row by row works, but then the recorded gain is no longer the common g that the coverage tests check. | `ReceiverModel`; generator |
| Frequency grids of `SParams`, `Source` and `ReceiverModel`. None carries `freqs_mhz`, so the § 2 check cannot run on them. Shape checks catch a different channel count but not a different grid of the same length, and a length-1 `SParams` broadcasts silently in `embed`. `_validate.same_grid` exists, but nothing calls it. Either add `freqs_mhz` to these objects, or build them only inside stages and adapters from objects whose grids are checked. | Every stage; `embed`, `power` |
| Batch axes in the forward model. § 2 promises `(..., n_time, n_freq)`, but `power`, `Source`, `ReceiverModel.gain` and `radiometer_noise` accept 2-D arrays only, so pushing `Posterior.draw` samples through `power` means a loop that rebuilds a `ReceiverModel` per draw. A closed-form Gaussian `predict` does not need batching; sampling Γ from `gamma_cov` does. Decide: batch these, or narrow § 2. | `predict`, § 10 held-out checks |
| Mapping `Reflection` epochs onto samples. `Reflection` holds `(n_meas, n_freq)` at measurement times, while `Source.gamma_s` and `ReceiverModel.gamma_rec` are per sample. Undecided: the time mapping (nearest, interpolation, or piecewise constant between changepoints), how `gamma_cov` propagates, and the prior for samples far from any measurement, which matters because S11 epochs can be days apart. | `fit_noise_waves`, `calibrate`, `predict`; adapters |

---

## 12. Changelog
- **v0, 2026-09-13:** first draft, as part of the Deployment 5 interface spec.
- **v0, 2026-09-14:** four gaps found while building the v0 forward model, added as open items; no interface change yet.
  - Drifting r_s against the fixed `(n_freq,)` `path_gain_ratio` (§ 4.2).
  - No frequency grid on `SParams`, `Source` or `ReceiverModel`, so § 2's grid check cannot run on them.
  - No batch axis in `power` and the objects it takes, against § 2.
  - No rule for mapping `Reflection` epochs onto samples; the `gamma_rec` note now says epochs are sparse.

  Sections changed: § 4.2, 11.
- **v0, 2026-09-14:** split out of that spec, with the same section numbers. The sections on eigsim, the D5 adapters and the generator's truth model (§ 7–9), and D5-specific notes elsewhere, stay in the D5 spec. No interface change.
- **v0, 2026-09-14:** equations written out (§ 4.1–4.3 and a new § 4.5), so the spec no longer depends on memo M003 being at hand. No interface change.

  Sections changed: § 1, 3, 4, 6, 10.
- **v0, 2026-09-14:** memo M003 now ships with the package as Markdown (`docs/memos/`), and the background links to it. No interface change.
