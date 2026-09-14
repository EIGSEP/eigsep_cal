# Interface spec v0: eigsep_cal, eigsim and the D5 adapters

*Status: draft, 2026-09-13. Owner: Christian Hellum Bye.*

This spec lets two tracks proceed in parallel:
- **track A:** eigsep_cal (instrument model and calibration stages);
- **track B:** eigsim's D5 observation mode.

Items marked **Open** need a decision before code depends on them. Once a consumer is built on this spec, every change bumps the version and gets a line in the changelog (§ 12).

**Background**
- The staged Bayesian framework was decided in the D5 analysis workspace (`~/Documents/research/papers/eigsep_analysis`, `docs/logbook.md`, 2026-09-13). Each stage passes a posterior, not a point estimate; there is no single joint fit.
- The physics and notation come from memo M003 (`manuscript/memos/M003_receiver_calibration/memo.tex`). Equation names below refer to its labels.

---

## 1. Components and dependency direction

```
 sky, beam, horizon                                D5 files (corr, S11, telemetry)
        │                                                        │
 eigsim (mock_analysis)                                  adapters (workspace code/
        │ plain arrays: t_ant                            or data-analysis)
        ▼                                                        │
 generator (mock_analysis) ──► Observation, Reflection ◄─────────┘
                                        │
                                        ▼
                                   eigsep_cal
                         forward model, stage solvers
                                        │
                                        ▼
                     Posterior, CalibratedSpectrum ──► stage 5 (sky spectra), rotis
```

**Rules**
- **eigsep_cal** depends on numpy; SciPy and JAX may be added when needed. It never reads instrument data files, and never models sky, beam or terrain. It knows nothing about D5 quirks. It may save and load its own objects (§ 5.6).
- **eigsim** never imports eigsep_cal and knows nothing about receivers. It returns plain arrays.
- **The generator** in mock_analysis is the only code that imports both (home: Q-CHB-29).
- **Adapters** do all file I/O and all D5 handling (§ 8). They emit the same objects as the generator, so a stage cannot tell synthetic input from real input.

---

## 2. Conventions

| Quantity | Convention |
|---|---|
| Frequency | `freqs_mhz`: float64, MHz, strictly increasing |
| Time | `times_unix`: float64, seconds since 1970-01-01 UTC. D5 uses `time_best`, never header `times`. eigsim takes Julian day, `jd = times_unix / 86400 + 2440587.5`; the generator converts. |
| Temperature | Kelvin, float64 |
| Power | Linear, arbitrary units (correlator counts for D5), float64. Never dB. |
| Reflection coefficient | complex128, 50 Ω reference, looking into the device from the reference plane P (§ 3) |
| Array axes | Leading batch axes, then time, then frequency: `(..., n_time, n_freq)` |
| Flags | bool; `True` means bad (hera convention) |
| Precision | float64 and complex128 throughout. eigsim enables JAX x64 on import. |

**Frequency grids**
- **No resampling inside eigsep_cal.** All arrays passed to one call share an identical `freqs_mhz` array, checked with `np.array_equal`. eigsep_cal never interpolates in frequency.
- **Resampling happens upstream.** For example, adapters map the 1000-point S11 grid onto correlator channels, and the input's uncertainty carries the interpolation error.
- **D5 channel grid:** `f_k = k * 250 / 1024` MHz for k = 0…1023, so Δν = 0.244140625 MHz (`eigsep_observing.utils.calc_freqs_dfreq`).
- **Synthetic D5 studies use a subset of these channels** and carry the integer channel indices (`chan`), so results compare with data channel by channel.
- eigsim's default grid (50–250 MHz in 1 MHz steps) is not a subset, so D5 mode passes the channel grid explicitly.

---

## 3. Reference plane and switch states

**Reference plane P** is the common LNA-side node of the switch network (M003 § Decisions, item 3).
- Every state's switch path is part of its source: Γ_s by embedding, T_s by eq. `availgain`.
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

**The antenna includes the balun and the balun–switch coax** (CHB, 2026-09-13)
- The switch comes after the coax from the antenna balun. The S11 chain and the in-situ calibration de-embed only switch paths, so for both of them the coax is part of the antenna. The measured Γ_ant includes the balun and coax, and so does the calibrated antenna temperature (§ 6).
- The beam models (HFSS, eigsim) are free space: no balun loss and no coax. eigsim's `t_ant_k` (§ 5.1) therefore sits at a different plane from anything eigsep_cal measures or calibrates.
- The coax was destroyed when the telescope fell at the end of D5, so its S-parameters can never be measured.
- **This is fine for calibration.** No D5 fit needs these S-parameters. It only makes comparisons with simulations harder: the generator (§ 4.3) and stage 5 must model the balun and coax, using priors instead of measurements.

**Traps and scope**
- The VNA key `load` is the noise diode **off** (`cmt_vna.VNA.measure_ant` switches to VNANOFF). The ambient load is `amb`.
- `RFSP2` is unused in D5.
- Rows labelled `VNA*`, `UNKNOWN` or `MISSING` never reach a stage. Adapters either resolve them to a state from the spectra or drop them, and record the count in `provenance`. v0 has no latent-state model (§ 11).

---

## 4. Physics in the forward model

### 4.1 Power per state
M003 eq. `Ps`, plus an additive post-switch term:

```
P_s(ν, t) = g_s(ν, t) · [ M_s T_s + |Γ_s F_s|² T_unc + Re(Γ_s F_s) T_cos
                           + Im(Γ_s F_s) T_sin + T_0 ] + A_s(ν, t)
```

- F_s and M_s follow eq. `F`: F_s = sqrt(1 − |Γ_rec|²) / (1 − Γ_s Γ_rec), and M_s is the mismatch factor.
- g_s = g · r_s, where r_s is a path-gain ratio. It is 1 by default; the generator can perturb it, since M003 needs stable ratios.
- A_s is zero in the v0 fit. The generator can switch it on; for example, the 1 MHz comb is state-independent, on box-air only.

### 4.2 Receiver parameters: `ReceiverModel`

| Field | Shape | Notes |
|---|---|---|
| `t_unc_k`, `t_cos_k`, `t_sin_k`, `t0_k` | `(n_freq,)` or `(n_time, n_freq)` | Rogers & Bowman parameterisation |
| `t_r_k`, `t_l_k`, `c_k` | same; `c_k` complex | Intrinsic parameterisation (M003 eq. `map`). Use either this set or the row above, selected by `parameterisation`. |
| `gamma_rec` | `(n_freq,)` or `(n_time, n_freq)` complex | Hourly in D5 |
| `gain` | `(n_time, n_freq)` | Must be > 0 |
| `path_gain_ratio` | `dict[state, (n_freq,)]` | Defaults to 1 |

In the generator, any of these may be a function of a named covariate from `Observation.covariates`, for example a temperature coefficient.

### 4.3 Source temperatures
- **Path embedding.** `embed(gamma_term, t_term_k, sparams, t_path_k) -> (gamma_s, t_s_k)`:
  - Γ is cascaded through the two-port;
  - T_s follows eq. `availgain`;
  - |S21|² is taken as |S12·S21|, which assumes the path is reciprocal and passive.
- **Noise source** (generator only).
  - Diode excess temperature: T_ex = 290 K · 10^(ENR/10).
  - Behind a matched pad with loss factor L ≥ 1 at physical temperature T_pad, the source temperature before the path is (T_diode + T_ex) / L + (1 − 1/L) · T_pad.
  - The D5 ENR and pad values are open (Q-CGT-05).
- **Fits never need this.** They use M003's effective constants T_NS and T_L (eq. `TNSTL`), with a prior on T_NS.
- **Balun and coax** (generator only).
  - The generator turns eigsim's free-space `t_ant_k` into the RFANT source with `embed`: first through a lossy balun two-port, then through the coax two-port at its physical temperature, then through the RFANT switch path. `gamma_term` is the free-space antenna reflection.
  - Neither two-port has a D5 measurement (§ 3). Draw their parameters from a prior (cable type and length, datasheet loss, balun loss, cable temperature) and record the draw in `provenance`.
  - Fits never need them, because the measured Γ_ant already includes both.

### 4.4 Noise
- **Radiometer noise.** Per sample, σ² = P² / (B_eff · τ · n_int).
- **Bandwidth.** `enbw_hz` defaults to Δν = 244 140.625 Hz, until the SNAP polyphase filterbank's equivalent noise bandwidth is measured (§ 11).
- **Channel correlation.** Channels are independent in v0.

---

## 5. Objects

These are dataclasses of numpy arrays, validated on construction (shapes, dtypes, identical frequency grids) and treated as immutable. Names are provisional until track A's first commit.

### 5.1 `SkyTemperature`
Built by the generator from eigsim output; consumed by the forward model.

| Field | Shape | Notes |
|---|---|---|
| `t_ant_k` | `(n_time, n_freq)` | Available noise temperature at the terminals of the free-space antenna model: everything eigsim models (sky, ground, horizon). **No receiver term, no balun loss, no coax.** The generator adds the balun and everything after it with eigsep_cal's `embed` (§ 4.3). This is not the plane of a calibrated spectrum (§ 3). |
| `freqs_mhz` | `(n_freq,)` | |
| `times_unix` | `(n_time,)` | |
| `antenna` | str | `"box-air"` or `"box-gnd"` |
| `elevation_deg`, `azimuth_deg` | `(n_time,)` | Drive angles per sample; provenance only |
| `meta` | dict | eigsim version, config, beam, horizon and sky model |

### 5.2 `SParams`

| Field | Shape | Notes |
|---|---|---|
| `s11`, `s12s21`, `s22` | `(n_freq,)` complex | Same layout as the rows of `switch_sparams.npz`: `[S11, S12·S21, S22]` |
| `name` | str | For example `"RFANT"` or `"VNAANT"` |

**Open:** which port faces P? Adapters must state it; the lab answer is Q-CGT-03.

### 5.3 `Reflection`
The output of stage 2 (S11 calibration), consumed by stage 3.

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
| `tau_s` | `(n_time,)` | Time per integration (D5: 0.268 s, then 0.537 s from Jul 15) |
| `n_int` | `(n_time,)` int | Integrations averaged into the sample |
| `flags` | `(n_time, n_freq)` bool | `True` = bad |
| `freqs_mhz` | `(n_freq,)` | |
| `chan` | `(n_freq,)` int or `None` | Correlator channel index |
| `antenna` | str | |
| `covariates` | `dict[str, (n_time,)]` | Physical temperatures in K, NaN where missing: `t_amb_k`, `t_switch_k` (§ 8) |
| `changepoints_unix` | `(n_cp,)` | Epochs where drift parameters may jump, taken from the D5 timeline |
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
# and changepoints, so the switched ratio Q_s (M003 eq. Q) and its
# uncertainty are available at every non-reference sample.

# Stage 3b: noise waves and reference constants.
fit_noise_waves(obs, reflections, post_3a, prior) -> Posterior
# Fits Θ = (T_unc, T_cos, T_sin, T_NS, T_L) plus drift coefficients
# from the calibrator states (M003 eq. cal / X), with an explicit T_NS prior.

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
- **`CalibratedSpectrum.t_ant_k` is not eigsim's `t_ant_k`.** It is the RFANT source temperature at P, covering the antenna, balun, coax and RFANT switch path (§ 3). Measured S-parameters can remove the switch path; nothing can remove the balun and coax. To compare with simulations, stage 5 forward-models them and marginalises over their priors (§ 4.3).
- 3a comes before 3b because the model is bilinear in gain and noise waves. The v0 validation (§ 10) must compare the two-stage result with a joint fit on synthetic data, to measure what the split costs.

---

## 7. eigsim deliverables (track B)

- **Path mode.** Add `simulate_path(beam_data, freqs_mhz, sky, times_jd, elevations_deg, azimuths_deg, config=None, ...) -> t_ant`, shaped `(n_time, n_freq)`.
  - It takes one orientation per time sample, as D5 motor telemetry gives.
  - It does **not** add `receiver.temperature`.
  - Group samples by unique (elevation, azimuth) and simulate each group's times together. D5 orientations repeat: static at night, a raster on Jul 17.
  - The existing `simulate()` and its grid output stay unchanged.
- **Rotation.** Keep `drive_rotation_matrix` for now. Isolate the composition so that the answer to Q-CHB-23 is a one-line change, not a refactor.
- **Two antennas.** box-air (suspended) and box-gnd (on the ground) each get their own beam, horizon and ground treatment from config (Q-CHB-28).
- **Frequencies.** Accept an arbitrary frequency array, in particular the D5 channel grid.
- **Output.** float64, documented as the `t_ant_k` of `SkyTemperature` (§ 5.1). It describes the free-space antenna; balun and coax effects belong to the generator, not eigsim (§ 3).
- **Tests.**
  - At matching (orientation, time) samples, path mode equals grid mode minus the receiver term.
  - Grouping by unique orientation gives the same result as running each sample separately.

---

## 8. What D5 adapters own

These never enter eigsep_cal:
- **Times.** `time_best` for correlator rows; `metadata_snapshot_unix` for S11 measurements.
- **Antenna mapping by wiring phase.** Phase C: key 4 = box-air, key 0 = box-gnd. The switched receiver is key 3 in phases A and B. `header/input_to_ant` is stale.
- **Row and channel problems.**
  - Dropped integrations (all-zero rows) are removed.
  - int32 wraps become channel flags.
  - The 1 MHz comb mask (`products/005_air_comb/air_comb_channel_mask.csv`) applies to box-air from Jul 15 19:16 to Jul 16 10:50 MDT, and to box-gnd at better than ~0.5 %.
- **Switch labels.** `MISSING`, `UNKNOWN` and `VNA*` rows (§ 3).
- **Integration time.** It changes on Jul 15, so set `tau_s` per row.
- **Covariates.**
  - `t_amb_k` comes from `tempctrl_load.T_now`. The reading counts, whatever the controller's drive state (Q-CHB-26).
  - `t_switch_k` comes from `rfswitch_therm`, after cleaning.
- **S11 inputs.**
  - Resample from the 1000-point grid, carrying the resampling error.
  - The OSL keys are Q-CGT-01; the switch-path file is Q-CGT-02 and Q-CGT-03.
  - Remember the `load` = RFNOFF trap.
  - Files with a singular OSL solve are Q-CHB-12.
- **Changepoints.** From `docs/field_notes/d5_timeline.md`: box-air lifts, reboots, battery failures, the comb onset.

---

## 9. Truth richer than the fit

The generator must be able to switch on each of these effects independently, even though the v0 fit omits them. Every test states which ones are on.
- Temperature-dependent noise waves and reference constants.
- The intrinsic parameterisation with a drifting measured Γ_rec(t).
- Drifting path-gain ratios r_s.
- Full SP1 cable S-parameters, with the ripple.
- The balun and balun–switch coax, with parameters drawn from a prior (§ 4.3). The fits see them only through Γ_ant.
- The 1 MHz comb as an additive A_s (box-air only, state-independent).
- int32 wrap clipping.
- Dropped integrations.
- Unresolved state labels.
- Leakage between neighbouring channels.

---

## 10. Validation contract (every stage)

- **Coverage.**
  - Run ≥ 200 synthetic realisations at D5 cadence: RFANT 600 s / RFNON 60 s / RFAMB 60 s, with D5 gaps and SP1 at only one epoch.
  - Per block, the fraction of truths inside the central 68 % and 95 % intervals must agree with nominal at the 99 % binomial level.
  - Report it once with the § 9 effects off and once with them on.
- **Prior against posterior.** Report the § 5.5 diagnostics for every block.
- **Held-out prediction.** Score `predict` against data kept out of the fit (one SP1 termination; next-night AMB/NON) as χ² against the predictive covariance.
- **Forward-model tests in CI.**
  - Port M003 notebook 003 section A into a unit test: eq. `Ps` against a direct wave-equation solution, to 1e-11 K.
  - Test eq. `map` with the factor of 2.

---

## 11. Open

| Item | Needed by | Tracked |
|---|---|---|
| Rotation composition of the motor mount | B | Q-CHB-23 |
| box-gnd beam, ground and horizon model | B | Q-CHB-28 |
| Home of the generator in mock_analysis, and how mock_analysis depends on the unpushed eigsep_cal | A + B integration | Q-CHB-29 |
| S-parameter port orientation and reference planes | A (`embed`), stage 2 | Q-CGT-03 |
| Noise-source pad and ENR | A (priors, generator) | Q-CGT-05 |
| Balun and balun–switch coax model and priors. The coax was destroyed, so D5 has no measurement. | Generator; stage 5 comparisons with simulations | Q-CGT-10; IMP-05 (future deployments) |
| SNAP channel equivalent noise bandwidth and neighbour correlation | A (noise model) | Analysis to-do, from the PFB taps in the firmware |
| Latent switch state for `MISSING` rows | v1 | — |
| Structured or sparse covariance for large blocks (dense above ~10⁴ parameters is too big) | v1 | — |

---

## 12. Changelog
- **v0, 2026-09-13:** first draft.
- **v0, 2026-09-13 (same day, before any consumer):** the RFANT source includes the balun and the unmeasurable balun–switch coax, while the beam models are free space (§ 3, 4.3, 5.1, 5.3, 6, 7, 9, 11).
