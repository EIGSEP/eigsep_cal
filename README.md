# eigsep_cal

EIGSEP instrument model and staged Bayesian calibration.

The package is being rebuilt. It will hold:

- an instrument forward model from antenna temperature to measured power
  per switch state (noise waves coupled through the reflection
  coefficients, noise source, gain drift, switch-path S-parameters,
  additive post-switch terms);
- stage solvers that return posteriors, not point estimates, so later
  stages can carry the uncertainty forward;
- nothing that models the sky. Sky, beam and horizon simulation lives in
  `eigsep_mock_analysis` (eigsim), which imports this package to generate
  synthetic data.

Today it contains the v0 forward model and data objects of
`docs/api.md`:

- `SParams`, `embed`: two-port cascade and available gain (§ 4.3);
- `ReceiverModel`, `noise_wave_map`: receiver noise parameters (§ 4.2);
- `Source`, `power`, `radiometer_noise`: power per switch state (§ 4.1,
  § 4.4), checked against a direct wave-equation solution to 1e-11 K;
- `SkyTemperature`, `Reflection`, `Observation`, with npz save/load for
  the last two (§ 5).

It also still contains `dicke.calc_Tant_star` (Monsalve et al. 2017,
eq. 1).

It also contains `calkit`, the OSL calibration math of Monsalve et al.
2016, moved here from `cmt_vna` so that package only drives the
instrument; and `S11`, below.

## S11

`S11` reads the raw S11 HDF5 files written by the observing VNA loop
(`eigsep_base.io.write_s11_file`) and calibrates them in process. It
replaces `data-analysis/scripts/calibrate_field_s11.py` and the
per-DUT `*_calibrated.h5` files it used to write; nothing is written
to disk now.

```python
from eigsep_cal import S11

s11 = S11(datadir)                     # a directory or a single file
s11.calibrate()                        # 2026 packaged characterizations
```

`calibrate()` defaults to the characterization files shipped in
`src/eigsep_cal/data/` for the object's season; see that directory's
README for their format and naming. 2026 is shipped; until 2025 is
added, that season needs its files passed explicitly:

```python
s11 = S11(datadir, year=2025)
s11.calibrate(switchpaths, osldata)
```

`year=None` has no switch S-parameters, so only the `raw` and `vna`
planes are produced, against an ideal OSL model. No characterization
files are needed and timestamps come from whichever source the file
has, making it the quickest look at a directory whose season you have
not established:

```python
S11(datadir, year=None).calibrate()
```

Loading and calibrating are separate: the constructor only reads
files, so the object is cheap to build and can be inspected (`duts`,
`timestamps`, `freqs`, `skipped`) before committing to a pair of
characterization files.

### Calibration chain

```
raw --[de-embed the VNA's own internal OSL]--> vna
    --[de-embed the VNA-leg switch path]-----> dut
    --[embed the RF/LNA-leg switch path]-----> lna
```

How far a DUT gets is a property of the season's switch topology, held
in the hard-coded `DEEMBED_DICTS` / `EMBED_DICTS` and picked by `year`
(`None` has no switch S-parameters, so everything stops at `vna`).
A DUT with an empty list already sits at that plane and stops one
shallower: in 2026 `load` and `noise` stop at `vna`, and `rec` stops at
`dut`, while 2026 `ant`/`amb` and 2025 `load`/`noise` run the whole
chain. Every capture also carries a `"default"` alias pointing at the
deepest plane it actually reached.

`year` also selects where capture timestamps come from. 2026 headers
carry `metadata_snapshot_unix`; 2025 files have no such key, so the
time is parsed from the filename (`ants11_20250719_102527.h5`) as
Mountain Standard Time, a fixed UTC-7.

Passing `None` for both arguments, or building with `year=None`, uses
an ideal (+1, -1, 0) OSL model and reaches `vna` only -- the
quick-look path, needing no characterization files. This replaces the
old `RawS11`.

### Reading the result

The result is indexed by DUT, capture time and plane. Fix two, sweep
the third:

```python
times, s11s  = s11.get_all_s11s(dut, plane)        # sweep time
times, s11s  = s11.get_all_duts(timestamp, plane)  # sweep DUT
time,  planes = s11.get_all_planes(dut, timestamp) # sweep plane
```

Each returns the matched capture time(s) alongside the data, because a
requested timestamp snaps to the nearest capture and the DUTs are not
measured in lockstep -- the receiver sits in its own files, so
`get_all_duts` matches each DUT separately and returns a `{dut: time}`
dict. A DUT or capture that never reached the requested plane is left
out rather than raising. `get_s11(dut, timestamp)` additionally pairs
the DUT with the receiver, and `get_dly` / `get_all_dlys` give the
Blackman-Harris-windowed delay spectrum on `dlys`.

### Frequency axes

Nothing here interpolates in frequency. A directory can mix sweep
spans -- a run out to 500 MHz beside the usual 250 MHz ones -- so the
axis shared by most files wins and the rest are excluded, each listed
in `skipped` with a reason.

### eigsep_base

Reading raw files needs `eigsep_base`, which is not on PyPI. It is
imported lazily, inside the reader, so the rest of the package stays
installable and importable from PyPI dependencies alone. Install it
from a sibling checkout:

```bash
pip install -e ../eigsep_base
```

The 2024 Vivaldi gain calibration (`vivaldi_cal/`) and the `eigsep_corr`
dependency were removed; they are preserved at tag `legacy-2024`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```
