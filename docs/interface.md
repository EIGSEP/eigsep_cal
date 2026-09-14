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
- **eigsep_cal** depends on numpy. The v0 stage solvers are closed-form Gaussian (CHB, 2026-09-13): the drift model of 3a is linear in its basis coefficients, and given Q_s the calibration equation of 3b is linear in Θ. SciPy, JAX or a sampler come in only when a nonlinear block needs them. It never reads instrument data files, and never models sky, beam or terrain. It knows nothing about D5 quirks. It may save and load its own objects (§ 5.6).
- **eigsim** never imports eigsep_cal and knows nothing about receivers. It returns plain arrays.
- **The generator** in mock_analysis is the only code that imports both. It cannot live inside eigsim, which never imports eigsep_cal. It is its own mock_analysis workspace member (Q-CHB-29, CHB 2026-09-13).
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
- eigsim's default grid (config `eigsep`) is the 52 channels its beam was simulated at, k = 192, 208, …, 1008, so it is a subset. The `eigsep_1mhz` grid (50–246 MHz, 1 MHz steps) is not, and the pinned `eigsep_v000` grid is not either.

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
  - The D5 ENR and pad values are open (Q-CGT-05). Until Charlie answers, use eigsep_observing v2.14.0 `obs_config.yaml` as a stand-in: ENR 35 dB behind a 30 dB pad, so T_ex / L ≈ 917 K at the switch. The instrument paper's 31 dB behind −20 dB conflicts with it.
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
| `t_ant_k` | `(n_time, n_freq)` | Available noise temperature at the terminals of the free-space antenna model: everything eigsim models, meaning sky, horizon, and ground at the level recorded in `meta` (§ 7.1). **No receiver term, no balun loss, no coax.** The generator adds the balun and everything after it with eigsep_cal's `embed` (§ 4.3). This is not the plane of a calibrated spectrum (§ 3). |
| `freqs_mhz` | `(n_freq,)` | |
| `times_unix` | `(n_time,)` | |
| `antenna` | str | `"box-air"` or `"box-gnd"` |
| `elevation_deg`, `azimuth_deg` | `(n_time,)` | Drive angles per sample; provenance only |
| `meta` | dict | eigsim version, config, beam, horizon and sky model; `ground_level` and every drawn ground parameter (§ 7.1) |

### 5.2 `SParams`

| Field | Shape | Notes |
|---|---|---|
| `s11`, `s12s21`, `s22` | `(n_freq,)` complex | Same layout as the rows of `switch_sparams.npz`: `[S11, S12·S21, S22]` |
| `name` | str | For example `"RFANT"` or `"VNAANT"` |

**Port convention:** cmt_vna `calkit`'s, which data-analysis's S11 chain (`scripts/calibrate_field_s11.py`) uses.
- Γ' = S11 + S12·S21 Γ / (1 − S22 Γ): port 1 (`s11`) faces the reference side, and port 2 (`s22`) faces the termination.
- The reference side is P for `RF*` paths and the VNA for `VNA*` paths.
- Whether the lab measurements in `switch_sparams.npz` were taken in that orientation is Q-CGT-03.

### 5.3 `Reflection`
The output of stage 2 (S11 calibration), consumed by stage 3. data-analysis owns stage 2, and its D5 adapter builds this object (§ 6).

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

**Stage 2 (S11 calibration) lives in data-analysis, not in eigsep_cal** (CHB, 2026-09-13; Q-CHB-31).
- `scripts/calibrate_field_s11.py` already runs the chain with cmt_vna `calkit`: raw → internal open/short/load (`vna` plane) → de-embed the `VNA*` path (`dut`) → embed the `RF*` path (`lna`, which is P).
- Keeping one implementation matters: Q-CGT-04 shows two versions of this calibration disagreeing by \|ΔΓ\| ≈ 0.06–0.3.
- The chain gives point estimates. The D5 adapter wraps the `lna` plane as `Reflection` and supplies `gamma_cov` from two sources: the scatter between repeat captures, and Monte Carlo through the same chain with perturbed inputs (standard models, switch-path S-parameters, resampling onto correlator channels).
- eigsep_cal's `embed` repeats the Γ cascade, because the forward model also needs T_s. The generator package tests it against `calkit.embed_sparams`, so the two cannot drift apart.

---

## 7. eigsim deliverables (track B)

- **Path mode.** Add `simulate_path(beam_data, freqs_mhz, sky, times_jd, elevations_deg, azimuths_deg, config=None, ...) -> t_ant`, shaped `(n_time, n_freq)`.
  - It takes one orientation per time sample, as D5 motor telemetry gives.
  - It does **not** add `receiver.temperature`.
  - Group samples by unique (elevation, azimuth) and simulate each group's times together. D5 orientations repeat: static at night, a raster on Jul 17.
  - The existing `simulate()` and its grid output stay unchanged.
- **Rotation.** `drive_rotation_matrix` (`eigsim/rotations.py`) composes `R_X(el) @ R_Z(az)`, applied body→top.
  - That is already the physical azimuth-outer mount (Aaron's top→body `R_Z(−az) R_X(−el)`), which gives the transmitter 2-D nadir coverage. Over the D5 raster grid it reaches 768/768 nside-8 pixels, against 55/768 for `R_Z(az) R_X(el)` (workspace logbook 2026-09-13; Q-CHB-23).
  - **Do not flip it.** Stage 4 still confirms the composition from the Jul 17 raster (workspace roadmap § 4).
  - Keep the composition in one place, so any change after that check is one line.
- **Two antennas.** box-air (suspended) and box-gnd (on the ground) each get their own beam, horizon and ground-model level (§ 7.1) from config (Q-CHB-28).
  - **box-gnd** sits directly on the ground: the bowtie on a box identical to box-air's, then soil, with **no ground plane**.
  - Its orientation never changed during D5; the value is Q-ARP-01.
  - Start from the HFSS free-space bowtie beam, the best model available. Ground coupling is not simulated (G4, § 7.1).
- **Frequencies.** Accept an arbitrary frequency array, in particular the D5 channel grid.
- **Output.** float64, documented as the `t_ant_k` of `SkyTemperature` (§ 5.1). It describes the free-space antenna; balun and coax effects belong to the generator, not eigsim (§ 3).
- **Tests.**
  - At matching (orientation, time) samples, path mode equals grid mode minus the receiver term.
  - Grouping by unique orientation gives the same result as running each sample separately.

### 7.1 Ground model levels

eigsim today (G0) treats every direction below the horizon as a uniform blackbody: `T_ant = T_sky,above + fgnd · Tgnd`, with one `ground.temperature` of 300 K for all times, frequencies, directions and antennas (`eigsim/simulate.py`). croissant supports only a constant `Tgnd` and includes no scattering of the sky by the ground.

| Level | What directions below the horizon contribute | Reflection ripple | Needs |
|---|---|---|---|
| G0 | Uniform blackbody at one `Tgnd`; emissivity 1; no reflection. Current eigsim. | none | — |
| G1 | G0, with `Tgnd` drawn from a prior, optionally a function of time, and set per antenna | none | A `Tgnd` prior: D5 recorded no ground temperature (IMP-37) |
| G2 | Incoherent soil: emission (1 − \|R\|²) · T_phys, plus reflected sky power \|R\|² · T_sky from the specular direction. R depends on incidence angle, polarisation and soil. | none | Oblique-incidence, two-polarisation reflectivity (eigsep_terrain's `reflectivity.py` is normal incidence only); facet normals from the DEM |
| G3 | Coherent terrain multipath: direct and reflected waves from the same sky direction add as voltages, with each terrain pixel's excess delay | yes | The complex far-field pattern, `beam_cart` (below); eigsim's beam file holds power only (`bm`, float64). Per-pixel terrain distance (`horizon_mwss.npz`) and ray tracing (eigsep_terrain) |
| G4 | box-gnd only: soil in the antenna's near field changes the beam and its efficiency | — | EM simulation of bowtie, box and soil |

- **G0 to G3 are cumulative.** G4 replaces box-gnd's free-space beam.
- **Only G3 adds reflection ripple.**
  - A wave from sky direction n reaches the antenna directly, and again after reflection, from a direction n′ below the horizon with excess delay τ.
  - The response is |F(n) + R F(n′) e^{−2πiντ}|² T_sky(n), where F is the complex far-field pattern. Only the cross term ripples in frequency.
  - A power beam cannot represent the cross term; G2 keeps only |R F(n′)|².
- **Beam for G3** (found 2026-09-13).
  - `data-analysis/hfss_beam_maps/bowtie_beam.npz` holds `beam_cart`: complex Cartesian (Ex, Ey, Ez) in mV at 1 W incident power, plus `gain_th` and `gain_ph`. `beam_cart` is the same array as Dominic's `bowtie_beams_cart.npz`, which lacks the frequency and nside keys.
  - Grid: HEALPix nside 32 (about 1.8°); 52 frequencies from 46.875 to 246.09 MHz, 3.906 MHz apart (every 16th correlator channel). These are the transmitter frequencies: Bahram simulated them so Dominic could compare the beam with field measurements (CHB).
  - It is eigsim's default beam since 2026-09-13 (v001, § 11).
  - \|E\|² matches the shape of `gain_th + gain_ph` to better than 0.9 % of the peak at every frequency. Their ratio varies with frequency (1.96 at 47 MHz, 1.01–1.20 above), so use one representation, never a mix.
  - The integral of `tot_g` (gain from \|E\|²) stays below 4π and roughly tracks the Oct 2025 bowtie mismatch loss, as a realized gain should. E is transverse to within 0.7 %.
  - It is the same beam as BK's Oct 2025 bowtie-on-box simulation behind the instrument paper's `beam_maps.npz`: pattern correlation 0.999, 1.000 and 0.986 at 50, 150 and 250 MHz.
- **Flat-floor intuition.** For a floor a height h below the antenna, τ = 2h cos θ / c for a source at zenith angle θ. That is about 587 ns at h = 88 m and θ = 0, and it falls toward grazing, where the finite floor and the canyon walls set the geometry instead. Reflections therefore spread over a range of delays, not a single spike.
- **Record the level.** `SkyTemperature.meta` carries `ground_level` and every drawn ground parameter (`Tgnd`, soil type, eps_r, resistivity).
- **`correct_ground_loss` holds only at G0 and G1.** From G2 on, the ground term depends on the sky, so it is no longer `fgnd · Tgnd`.
- **Stages 3a and 3b never see the ground.** RFANT enters them as data, so § 9 and § 10 do not vary the level. The level matters for stage 5 and for § 7.2.
- **Order.** Path mode ships at G0. G1 is needed before stage 5 compares simulations with data, and G3 before § 7.2.

### 7.2 Delay-filter check (box-air)

**Hypothesis** (CHB, 2026-09-13): box-air's terrain reflections can be removed with a delay filter, leaving spectra that a G0 or G1 model describes. box-gnd is not expected to pass (G4) and is not part of the check.

**Method** (a mock_analysis study; no receiver and no generator)
- Simulate box-air in path mode at G3 and at G2, with the same sky, beam, horizon, orientations, times and soil draw.
- Use the D5 channel grid (§ 2). Its alias-free delay limit is 1/(2Δν) ≈ 2048 ns. eigsim's default 1 MHz grid stops at 500 ns, below the ≈ 587 ns floor reflection.
- Apply the stage 5 delay filter (hera_filters DPSS or DAYENU) to both, with the data's flags, including the comb-mask gaps. "Filtered" below means the component the filter keeps.
- Report, as a function of delay cutoff, LST, orientation and height (≈ 88 m, and lower on Jul 15, Q-CHB-05):
  - **reflection leakage:** filtered G3 − filtered G2, the reflection that survives the filter;
  - **filter damage:** G2 − filtered G2, the structure the filter removes from a spectrum without reflections.
- Repeat over soil draws, since the D5 soil parameters are unknown.

**Outcome**
- **Pass** (both below the threshold, § 11): stage 5 compares filtered box-air data with filtered G1 simulations.
- **Fail:** stage 5 needs the G3 forward model, marginalised over soil priors.

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
  - Run stage 2 (`calibrate_field_s11.py`) and build `Reflection`, including `gamma_cov` (§ 6).
  - Resample from the 1000-point grid, carrying the resampling error.
  - The OSL keys are Q-CGT-01; the switch-path file is Q-CGT-02 and Q-CGT-03.
  - Remember the `load` = RFNOFF trap.
  - Files with a singular OSL solve are Q-CGT-09. The likely cause is zero-filled sweeps at session start (cmt_vna #54).
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
| Rotation composition of the motor mount. eigsim already implements the azimuth-outer mount, so do not flip it; stage 4 confirms it from the Jul 17 raster (§ 7) | Stage 4 | Q-CHB-23 (answered) |
| box-gnd orientation value (setup settled, § 7) | B | Q-ARP-01 |
| **Resolved (CHB, 2026-09-13):** the generator is its own mock_analysis workspace member. `rebuild` is on `EIGSEP/eigsep_cal`, so mock_analysis can take it as a git source. | A + B integration | Q-CHB-29 |
| S-parameter port orientation and reference planes | A (`embed`), stage 2 | Q-CGT-03 |
| Noise-source pad and ENR. Stand-in until answered: `obs_config.yaml`, ENR 35 dB behind a 30 dB pad (§ 4.3). | A (priors, generator) | Q-CGT-05 |
| SP1 cable type and length. Its temperature was not logged (CHB), so the generator and fits use a prior. | Generator (§ 9 SP1 S-parameters); § 10 held-out SP1 check | Q-CGT-06 |
| Balun and balun–switch coax model and priors. The coax was destroyed, so D5 has no measurement. | Generator; stage 5 comparisons with simulations | Q-CGT-10; IMP-05 (future deployments) |
| `Tgnd` prior, and whether it varies with time. D5 recorded no ground temperature. | B (G1) | IMP-37 |
| Marjum soil parameters (eps_r, resistivity) and terrain types per DEM facet. eigsep_terrain's `TERRAIN_TYPES` are generic. | B (G2, G3) | — |
| Phase reference point of `beam_cart`, which sets the G3 delays (§ 7.1) | B (G3) | With Bahram Khalichi; noted in Q-BK-01 |
| Interpolating `beam_cart`, phase included, from 3.906 MHz and nside 32 to the D5 channel grid and eigsim's resolution | B (G3) | — |
| **Resolved (CHB, 2026-09-13):** eigsim's beam. v000 differed from the Oct 2025 beam above about 150 MHz (pattern correlation 0.33 at 246 MHz). eigsim now defaults to v001: \|E\|² from `beam_cart`, normalised to directivity, on the 52 native channels (config `eigsep`), with a cubic-spline 1 MHz version (config `eigsep_1mhz`). The instrument paper's simulations (`horizon_position`, `horizon_chromaticity`) pin config `eigsep_v000`, so their figures reproduce; rerunning the paper with v001 is a separate task. mock_analysis branch `feat/eigsim-oct2025-beam`. | B | — |
| Which antenna position and height `horizon_mwss.npz` was computed for; horizons for box-air's D5 heights and for box-gnd | B | Q-CHB-05 (heights) |
| Delay cutoff and pass threshold for the delay-filter check, for example against radiometer noise or the calibration uncertainty of the averaged spectrum | Stage 5 (§ 7.2) | — |
| box-gnd soil-coupling EM simulation | B (G4) | — |
| SNAP channel equivalent noise bandwidth and neighbour correlation | A (noise model) | Analysis to-do, from the PFB taps in the firmware |
| Latent switch state for `MISSING` rows | v1 | — |
| Structured or sparse covariance for large blocks (dense above ~10⁴ parameters is too big) | v1 | — |

---

## 12. Changelog
- **v0, 2026-09-13:** first draft.
- **v0, 2026-09-13 (same day, before any consumer):** the RFANT source includes the balun and the unmeasurable balun–switch coax, while the beam models are free space (§ 3, 4.3, 5.1, 5.3, 6, 7, 9, 11).
- **v0, 2026-09-13 (same day, before any consumer):** CHB's answers recorded.
  - Mount believed azimuth-outer (Q-CHB-23).
  - box-gnd setup settled: on a box on soil, no ground plane, fixed orientation, HFSS free-space beam as the start (Q-CHB-28 → Q-ARP-01).
  - The generator cannot live in eigsim (Q-CHB-29).
  - The singular-OSL question is now Q-CGT-09.
  
  Sections changed: § 1, 7, 8, 11.
- **v0, 2026-09-13 (same day, before any consumer):** ground model levels G0–G4 (§ 7.1) and the box-air delay-filter check (§ 7.2), agreed with CHB.
  - eigsim today is G0: a uniform 300 K blackbody below the horizon, with no reflections.
  - Only coherent multipath (G3) produces reflection ripple, and it needs a complex beam.
  - `SkyTemperature.meta` records the level.
  - New open items: `Tgnd` prior, soil parameters, complex HFSS pattern, horizon positions, the check's threshold, box-gnd EM simulation.

  Sections changed: § 5.1, 7, 11.
- **v0, 2026-09-13 (same day, before any consumer):** complex beam for G3 located and checked (§ 7.1). eigsim's v000 beam disagrees with it above about 150 MHz. New open items: phase reference, interpolation, choice of the D5-mode beam.

  Sections changed: § 7.1, 11.
- **v0, 2026-09-13 (same day, before any consumer):** eigsim switched to the v001 beam, on native channels or 1 MHz, with the paper's studies pinned to v000. The default eigsim grid is now a subset of the D5 channels.

  Sections changed: § 2, 7.1, 11.
- **v0, 2026-09-13 (same day, before any consumer):** CHB's answers recorded.
  - The generator is its own mock_analysis workspace member (Q-CHB-29).
  - v0 stage solvers are closed-form Gaussian in NumPy.
  - The noise source uses `obs_config.yaml` as a stand-in: ENR 35 dB behind a 30 dB pad (Q-CGT-05).
  - The SP1 cable temperature was not logged (Q-CGT-06).
  - `SParams` adopt cmt_vna `calkit`'s port convention; Q-CGT-03 now only confirms the lab orientation.

  Sections changed: § 1, 4.3, 5.2, 11.
- **v0, 2026-09-13 (same day, before any consumer):** stage 2 (S11 calibration) belongs to data-analysis (Q-CHB-31). Its D5 adapter builds `Reflection` with a covariance from repeat-capture scatter and Monte Carlo through the same chain, and eigsep_cal's `embed` is tested against `calkit`.

  Sections changed: § 5.3, 6, 8.
- **v0, 2026-09-13 (same day, before any consumer):** eigsim's rotation is already the physical azimuth-outer mount, so § 7 no longer asks for a flip (workspace logbook 2026-09-13, Q-CHB-23). The beam phase-reference question is tracked in Q-BK-01.

  Sections changed: § 7, 11.
