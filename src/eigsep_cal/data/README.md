# Packaged characterization data

Lab characterizations of the instrument, shipped inside the package so
`S11.calibrate()` works with no arguments. `eigsep_cal.s11.PACKAGED_FILES`
maps a season to the files it uses; today only 2026 has them.

Files are named per season, so a new one drops in without code
changes:

| File | Role |
| --- | --- |
| `switch_sparams<year>.npz` | Switch-path S-parameters, de-embedded/embedded to reach the DUT and LNA planes |
| `calibrated_internal_osl<year>.npz` | Characterized OSL standards, the "true" gammas the VNA's internal OSL is solved against |

2026 is present. 2025 is not yet: `S11(..., year=2025).calibrate()`
raises `FileNotFoundError` naming the exact path to add it at.

Both must be on the same frequency axis as the raw data being
calibrated. Nothing here interpolates in frequency: a mismatch is an
error, not a resample.

## `switch_sparams<year>.npz`

One array per switch state, keyed by **state name**, each
`(3, n_freq)` complex — `s11`, `s12*s21`, `s22`, the convention
`calkit.de_embed_sparams` and `calkit.embed_sparams` take.

The names come from the per-season dictionaries in `s11.py`. For 2026:

```
VNAANT  VNAAMB  VNASP1  VNARF      # VNA leg, de-embedded
RFANT   RFAMB   RFSP1              # RF/LNA leg, embedded
```

2025 additionally used `VNAN` / `RFN` for the load and noise paths.
Only the states a given DUT needs must be present; a missing one is
reported by name when that DUT is calibrated.

Note these are *switch-state* names, not cable names. An older
`system_sparameters.npz` in `data-analysis` keys its arrays by cable
(`rf_cables`, `ns_cables`, ...) and is not a drop-in replacement.

## `calibrated_internal_osl<year>.npz`

Three arrays, `O`, `S` and `L`, each `(n_freq,)` complex — the
characterized reflection coefficients of the open, short and load
standards. A `freqs` array may be present and is ignored on load.

These are what the VNA's own `VNAO`/`VNAS`/`VNAL` captures are solved
against to build the one-port error model. Calibrating without them
(`S11(..., year=None)`, or `calibrate(None, None)`) falls back to an
ideal `+1, -1, 0` model, which is the quick-look path.

## Regenerating

These are lab measurements, not derived products — replacing them
changes every calibrated S11 downstream. Record what changed and when,
so a result can be traced back to the characterization it used.
