"""Tests for eigsep_cal.s11.

The raw-file reader lives in eigsep_base, which eigsep_cal imports
lazily and which need not be installed to run these. A fake
``eigsep_base.io`` is installed in ``sys.modules`` instead, returning
canned contents per path -- which also pins the reader contract these
tests rely on: ``read_s11_file(path)`` -> ``(data, cal_data, header,
metadata)``, with the ``cal:`` prefix already stripped off the
calibration keys.
"""

import sys
import types

import numpy as np
import pytest
from scipy.signal.windows import blackmanharris

from eigsep_cal import s11 as s11_mod
from eigsep_cal.s11 import S11, timestamp_from_filename

NFREQ = 8
FREQS = np.linspace(1e6, 250e6, NFREQ)
WIDE_FREQS = np.linspace(1e6, 500e6, 2 * NFREQ)

# A well-behaved internal-OSL set: nothing zero, so it is not culled
# as an unmeasured capture.
OSL = np.array(
    [
        np.full(NFREQ, 1.0 + 0j),
        np.full(NFREQ, -1.0 + 0j),
        np.full(NFREQ, 0.01 + 0j),
    ]
)

SWITCH_NAMES = ["VNAANT", "RFANT", "VNAAMB", "RFAMB", "VNARF", "VNAN", "RFN"]


@pytest.fixture
def fake_io(monkeypatch):
    """Install a fake eigsep_base.io whose reader serves a registry."""
    registry = {}

    def read_s11_file(fname):
        return registry[str(fname)]

    io = types.ModuleType("eigsep_base.io")
    io.read_s11_file = read_s11_file
    base = types.ModuleType("eigsep_base")
    base.io = io
    monkeypatch.setitem(sys.modules, "eigsep_base", base)
    monkeypatch.setitem(sys.modules, "eigsep_base.io", io)
    return registry


@pytest.fixture
def raw_dir(tmp_path, fake_io):
    """A factory writing placeholder files backed by the fake reader."""

    def add(
        name,
        data,
        *,
        mode="ant",
        timestamp=1000.0,
        freqs=FREQS,
        cal_data="default",
    ):
        path = tmp_path / name
        path.touch()  # real file, so globbing and is_file() work
        if cal_data == "default":
            cal_data = {
                "VNAO": OSL[0][: freqs.size],
                "VNAS": OSL[1][: freqs.size],
                "VNAL": OSL[2][: freqs.size],
            }
            if freqs.size != NFREQ:
                cal_data = {
                    k: np.resize(v, freqs.size) for k, v in cal_data.items()
                }
        header = {
            "mode": mode,
            "metadata_snapshot_unix": timestamp,
            "freqs": freqs,
            "npoints": freqs.size,
        }
        fake_io[str(path)] = (data, cal_data, header, {})
        return path

    add.dir = tmp_path
    return add


def _dut(value=0.3 + 0.1j, n=NFREQ):
    return np.full(n, value)


def _sparams_npz(tmp_path, names=SWITCH_NAMES, n=NFREQ):
    """Switch-path S-parameters shaped as calkit expects: (3, nfreq)."""
    path = tmp_path / "switchpaths.npz"
    arrays = {
        name: np.array(
            [
                np.full(n, 0.01 + 0j),  # s11
                np.full(n, 0.9 + 0j),  # s12 s21
                np.full(n, 0.02 + 0j),  # s22
            ]
        )
        for name in names
    }
    np.savez(path, **arrays)
    return path


def _osl_npz(tmp_path, n=NFREQ):
    path = tmp_path / "osl.npz"
    np.savez(
        path,
        freqs=np.linspace(1e6, 250e6, n),
        O=np.full(n, 1.0 + 0j),
        S=np.full(n, -1.0 + 0j),
        L=np.full(n, 0.01 + 0j),
    )
    return path


# -- loading --------------------------------------------------------


def test_init_does_not_calibrate(raw_dir):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()})
    s11 = S11(raw_dir.dir)

    assert s11.calibrated is False
    assert s11.duts == ["ant"]
    planes = next(iter(s11.ant.values()))
    assert set(planes) == {"raw"}
    np.testing.assert_array_equal(s11.freqs, FREQS)


def test_accepts_a_single_file(raw_dir):
    path = raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()})
    raw_dir("ants11_20260101_000100Z.h5", {"ant": _dut()}, timestamp=2000.0)

    one = S11(path)
    both = S11(raw_dir.dir)

    assert len(one.timestamps) == 1
    assert len(both.timestamps) == 2


def test_missing_path_is_an_error(tmp_path, fake_io):
    with pytest.raises(ValueError, match="no such file or directory"):
        S11(tmp_path / "nope")


def test_empty_directory_is_an_error(tmp_path, fake_io):
    with pytest.raises(ValueError, match="no files matching"):
        S11(tmp_path)


def test_unknown_year_is_an_error(raw_dir):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()})
    with pytest.raises(ValueError, match="year must be None or one of"):
        S11(raw_dir.dir, year=2024)


def test_2025_reads_the_timestamp_from_the_filename(raw_dir):
    # header timestamp is deliberately wrong; 2025 files have none
    raw_dir(
        "ants11_20250719_102527.h5", {"ant": _dut()}, timestamp=float("nan")
    )
    s11 = S11(raw_dir.dir, year=2025)

    assert s11.timestamps == [timestamp_from_filename("x_20250719_102527.h5")]
    assert s11.timestamps == [1752945927.0]  # 2025-07-19 10:25:27 MST


def test_2026_reads_the_timestamp_from_the_header(raw_dir):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()}, timestamp=4242.0)
    assert S11(raw_dir.dir).timestamps == [4242.0]


def test_zeroed_captures_are_dropped(raw_dir):
    raw_dir(
        "ants11_20260101_000000Z.h5",
        {"ant": _dut(), "amb": np.zeros(NFREQ, dtype=complex)},
    )
    assert S11(raw_dir.dir).duts == ["ant"]


def test_file_without_internal_osl_is_noted(raw_dir):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()}, cal_data={})
    s11 = S11(raw_dir.dir)

    assert s11.osls["ant"] == {}
    assert "no internal-OSL data" in s11.skipped[0][1]


def test_wider_sweep_span_is_skipped(raw_dir):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()})
    raw_dir("ants11_20260101_000100Z.h5", {"ant": _dut()}, timestamp=2000.0)
    raw_dir(
        "ants11_20260101_000200Z.h5",
        {"ant": _dut(n=2 * NFREQ)},
        timestamp=3000.0,
        freqs=WIDE_FREQS,
    )
    s11 = S11(raw_dir.dir)

    np.testing.assert_array_equal(s11.freqs, FREQS)
    assert len(s11.timestamps) == 2
    assert 3000.0 not in s11.timestamps
    assert len(s11.skipped) == 1
    assert "500 MHz" in s11.skipped[0][1]


# -- calibration ----------------------------------------------------


@pytest.fixture
def loaded(raw_dir):
    """One ant-mode file with four DUTs, plus a rec-mode file."""
    raw_dir(
        "ants11_20260101_000000Z.h5",
        {
            "ant": _dut(0.30 + 0.10j),
            "amb": _dut(0.20 + 0.05j),
            "load": _dut(0.02 + 0.01j),
            "noise": _dut(0.04 + 0.02j),
        },
    )
    raw_dir(
        "recs11_20260101_000100Z.h5",
        {"rec": _dut(0.15 + 0.03j)},
        mode="rec",
        timestamp=1100.0,
    )
    return raw_dir.dir


def test_full_chain_depth_per_dut(loaded, tmp_path):
    s11 = S11(loaded)
    s11.calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))

    assert s11.calibrated is True
    # ant/amb have both legs -> all the way to the LNA plane
    for dut in ("ant", "amb"):
        planes = next(iter(getattr(s11, dut).values()))
        assert set(planes) == {"raw", "vna", "dut", "lna", "default"}
    # load/noise sit at the VNA plane in 2026 -- both lists empty
    for dut in ("load", "noise"):
        planes = next(iter(getattr(s11, dut).values()))
        assert set(planes) == {"raw", "vna", "default"}
    # rec de-embeds its VNA leg but has no RF leg to embed
    planes = next(iter(s11.rec.values()))
    assert set(planes) == {"raw", "vna", "dut", "default"}


def test_default_alias_points_at_the_deepest_plane(loaded, tmp_path):
    s11 = S11(loaded)
    s11.calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))

    ant = next(iter(s11.ant.values()))
    np.testing.assert_array_equal(ant["default"], ant["lna"])
    load = next(iter(s11.load.values()))
    np.testing.assert_array_equal(load["default"], load["vna"])
    rec = next(iter(s11.rec.values()))
    np.testing.assert_array_equal(rec["default"], rec["dut"])


def test_2025_load_and_noise_reach_the_lna_plane(raw_dir, tmp_path):
    """2025 wired load/noise through VNAN/RFN, unlike 2026."""
    raw_dir(
        "ants11_20250719_102527.h5",
        {"ant": _dut(), "load": _dut(0.02 + 0.01j)},
    )
    raw_dir(
        "recs11_20250719_102600.h5", {"rec": _dut(0.15 + 0.03j)}, mode="rec"
    )
    s11 = S11(raw_dir.dir, year=2025)
    s11.calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))

    planes = next(iter(s11.load.values()))
    assert set(planes) == {"raw", "vna", "dut", "lna", "default"}


def test_quicklook_needs_no_files_and_stops_at_vna(loaded):
    s11 = S11(loaded)
    s11.calibrate(None, None)

    assert s11.calibrated is True
    planes = next(iter(s11.ant.values()))
    assert set(planes) == {"raw", "vna", "default"}
    np.testing.assert_array_equal(planes["default"], planes["vna"])


def test_quicklook_matches_an_explicit_ideal_osl(loaded, tmp_path):
    """The no-argument path is the ideal (+1, -1, 0) model."""
    ideal = tmp_path / "ideal.npz"
    np.savez(
        ideal,
        O=np.full(NFREQ, 1.0 + 0j),
        S=np.full(NFREQ, -1.0 + 0j),
        L=np.zeros(NFREQ, dtype=complex),
    )
    quick = S11(loaded).calibrate(None, None)
    explicit = S11(loaded).calibrate(_sparams_npz(tmp_path), ideal)

    a = next(iter(quick.ant.values()))["vna"]
    b = next(iter(explicit.ant.values()))["vna"]
    np.testing.assert_allclose(a, b)


def test_calibrate_returns_self_for_chaining(loaded):
    s11 = S11(loaded)
    assert s11.calibrate(None, None) is s11


def test_osl_model_on_a_different_grid_is_refused(loaded, tmp_path):
    s11 = S11(loaded)
    with pytest.raises(ValueError, match="never interpolates"):
        s11.calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path, n=2 * NFREQ))


def test_osl_model_missing_a_standard_is_refused(loaded, tmp_path):
    bad = tmp_path / "bad_osl.npz"
    np.savez(bad, freqs=FREQS, O=np.ones(NFREQ), S=-np.ones(NFREQ))
    s11 = S11(loaded)
    with pytest.raises(ValueError, match="missing key"):
        s11.calibrate(_sparams_npz(tmp_path), bad)


def test_missing_switch_path_names_the_file(loaded, tmp_path):
    sparams = _sparams_npz(tmp_path, names=["VNARF"])
    s11 = S11(loaded)
    with pytest.raises(ValueError, match="VNAANT.*is not in"):
        s11.calibrate(sparams, _osl_npz(tmp_path))


def test_no_internal_osl_for_a_mode_is_refused(raw_dir, tmp_path):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()}, cal_data={})
    s11 = S11(raw_dir.dir)
    with pytest.raises(ValueError, match="no usable internal-OSL"):
        s11.calibrate(None, None)


def test_nearest_in_time_osl_is_used(raw_dir, tmp_path):
    """Each capture takes the internal-OSL set closest in time."""
    raw_dir("ants11_a.h5", {"ant": _dut()}, timestamp=0.0)
    raw_dir(
        "ants11_b.h5",
        {"ant": _dut()},
        timestamp=1000.0,
        cal_data={
            "VNAO": np.full(NFREQ, 0.5 + 0j),
            "VNAS": np.full(NFREQ, -0.5 + 0j),
            "VNAL": np.full(NFREQ, 0.02 + 0j),
        },
    )
    s11 = S11(raw_dir.dir).calibrate(None, None)

    # Different OSL sets -> different "vna" planes for equal raw input
    assert not np.allclose(s11.ant[0.0]["vna"], s11.ant[1000.0]["vna"])


# -- packaged characterization files --------------------------------


def test_calibrate_defaults_to_the_packaged_files(
    loaded, tmp_path, monkeypatch
):
    """No arguments -> the season's shipped files are used."""
    sw, osl = _sparams_npz(tmp_path), _osl_npz(tmp_path)
    monkeypatch.setitem(
        s11_mod.PACKAGED_FILES,
        2026,
        {"switchpaths": sw, "osldata": osl},
    )
    default = S11(loaded).calibrate()
    explicit = S11(loaded).calibrate(sw, osl)

    a = next(iter(default.ant.values()))
    b = next(iter(explicit.ant.values()))
    assert set(a) == {"raw", "vna", "dut", "lna", "default"}
    np.testing.assert_array_equal(a["lna"], b["lna"])


def test_missing_packaged_files_name_the_path(loaded, tmp_path, monkeypatch):
    monkeypatch.setitem(
        s11_mod.PACKAGED_FILES,
        2026,
        {
            "switchpaths": tmp_path / "switch_sparams.npz",
            "osldata": tmp_path / "calibrated_internal_osls.npz",
        },
    )
    with pytest.raises(FileNotFoundError, match="switch_sparams.npz"):
        S11(loaded).calibrate()


def test_2025_packaged_files_are_not_shipped_yet(raw_dir):
    """Until they are added, the error names the path to add them at."""
    raw_dir("ants11_20250719_102527.h5", {"ant": _dut()})
    s11 = S11(raw_dir.dir, year=2025)
    with pytest.raises(FileNotFoundError, match="switch_sparams2025.npz"):
        s11.calibrate()


def test_explicit_none_still_means_quicklook(loaded):
    s11 = S11(loaded).calibrate(None, None)

    planes = next(iter(s11.ant.values()))
    assert set(planes) == {"raw", "vna", "default"}


def test_mixing_packaged_with_explicit_is_an_error(loaded, tmp_path):
    s11 = S11(loaded)
    with pytest.raises(ValueError, match="both default to the packaged"):
        s11.calibrate(switchpaths=_sparams_npz(tmp_path))
    with pytest.raises(ValueError, match="both default to the packaged"):
        s11.calibrate(osldata=_osl_npz(tmp_path))


def test_one_explicit_file_without_the_other_is_an_error(loaded, tmp_path):
    s11 = S11(loaded)
    with pytest.raises(ValueError, match="or None for both"):
        s11.calibrate(_sparams_npz(tmp_path), None)
    with pytest.raises(ValueError, match="or None for both"):
        s11.calibrate(None, _osl_npz(tmp_path))


def test_packaged_paths_point_into_the_package(tmp_path):
    """The shipped names, beside the module, are what data/README
    documents."""
    paths = s11_mod.PACKAGED_FILES[2026]
    assert paths["switchpaths"].name == "switch_sparams2026.npz"
    assert paths["osldata"].name == "calibrated_internal_osl2026.npz"
    assert paths["switchpaths"].parent == s11_mod.DATA_DIR
    assert s11_mod.DATA_DIR.is_dir()
    assert (s11_mod.DATA_DIR / "README.md").is_file()


def test_shipped_2026_files_load_and_match_the_dicts():
    """The files actually in the package are usable as they stand."""
    paths = s11_mod.PACKAGED_FILES[2026]
    if not paths["switchpaths"].is_file():
        pytest.skip("2026 characterization files not installed")

    osl = np.load(paths["osldata"])
    assert {"O", "S", "L"} <= set(osl.files)
    n = osl["O"].size

    sparams = np.load(paths["switchpaths"])
    needed = {
        name
        for d in (s11_mod.DEEMBED_DICT26, s11_mod.EMBED_DICT26)
        for names in d.values()
        for name in names
    }
    assert needed <= set(sparams.files), needed - set(sparams.files)
    for name in needed:
        assert sparams[name].shape == (3, n)


# -- year=None: no switch S-parameters ------------------------------


def test_year_none_calibrates_against_the_ideal_osl(loaded):
    """No arguments needed, and the chain stops at "vna"."""
    s11 = S11(loaded, year=None).calibrate()

    assert s11.calibrated is True
    for dut in s11.duts:
        planes = next(iter(getattr(s11, dut).values()))
        assert set(planes) == {"raw", "vna", "default"}


def test_year_none_matches_an_explicit_quicklook(loaded):
    ideal = S11(loaded, year=None).calibrate()
    quick = S11(loaded, year=2026).calibrate(None, None)

    np.testing.assert_array_equal(
        next(iter(ideal.ant.values()))["vna"],
        next(iter(quick.ant.values()))["vna"],
    )


def test_year_none_refuses_characterization_files(loaded, tmp_path):
    s11 = S11(loaded, year=None)
    with pytest.raises(
        ValueError, match="no switch S-parameters for year=None"
    ):
        s11.calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))


def test_year_none_takes_timestamps_from_either_source(raw_dir):
    """Works on 2026 files (header) and 2025 ones (filename)."""
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()}, timestamp=4242.0)
    from_header = S11(raw_dir.dir, year=None)
    assert from_header.timestamps == [4242.0]


def test_year_none_falls_back_to_the_filename(raw_dir, fake_io):
    """A 2025-style file has no header timestamp at all."""
    path = raw_dir("ants11_20250719_102527.h5", {"ant": _dut()})
    data, cal_data, hdr, meta = fake_io[str(path)]
    hdr.pop("metadata_snapshot_unix")

    s11 = S11(path, year=None)
    assert s11.timestamps == [1752945927.0]


def test_year_none_is_not_the_default(loaded):
    """2026 stays the default; None must be asked for."""
    assert S11(loaded).year == 2026


# -- lookups --------------------------------------------------------


def test_get_s11_pairs_the_dut_with_the_receiver(loaded, tmp_path):
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))
    sample = s11.get_s11("ant", 1000.0)

    assert sample.dut == "ant"
    assert sample.dut_time == 1000.0
    assert sample.rec_time == 1100.0
    np.testing.assert_array_equal(sample.dut_s11, s11.ant[1000.0]["default"])
    np.testing.assert_array_equal(sample.rec_s11, s11.rec[1100.0]["default"])


def test_get_s11_on_a_plane_the_capture_lacks_says_so(loaded):
    s11 = S11(loaded).calibrate(None, None)
    with pytest.raises(KeyError, match="no 'lna' plane"):
        s11.get_s11("ant", 1000.0, plane="lna")


def test_get_s11_for_an_unknown_dut_lists_what_loaded(loaded):
    s11 = S11(loaded).calibrate(None, None)
    with pytest.raises(ValueError, match="loaded DUTs are"):
        s11.get_s11("nope", 1000.0)


def test_get_all_s11s_is_time_ordered(raw_dir, tmp_path):
    for i, ts in enumerate([3000.0, 1000.0, 2000.0]):
        raw_dir(f"ants11_{i}.h5", {"ant": _dut(0.1 * (i + 1))}, timestamp=ts)
    s11 = S11(raw_dir.dir).calibrate(None, None)
    times, s11s = s11.get_all_s11s("ant", "vna")

    np.testing.assert_array_equal(times, [1000.0, 2000.0, 3000.0])
    assert s11s.shape == (3, NFREQ)


def test_get_all_s11s_skips_captures_without_that_plane(loaded, tmp_path):
    """A plane only some captures reached returns just those.

    This is what used to raise KeyError('default') when a raw-only
    capture sat beside calibrated ones.
    """
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))
    # "load" stops at vna, so it has no lna plane at any time
    times, s11s = s11.get_all_s11s("load", "lna")

    assert times.size == 0
    assert s11s.size == 0
    # ... while its default plane is there
    times, s11s = s11.get_all_s11s("load", "default")
    assert times.size == 1


def test_get_all_planes_sweeps_the_plane_axis(loaded, tmp_path):
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))
    time, planes = s11.get_all_planes("ant", 1000.0)

    assert time == 1000.0
    assert set(planes) == {"raw", "vna", "dut", "lna", "default"}
    np.testing.assert_array_equal(planes["default"], planes["lna"])
    for arr in planes.values():
        assert arr.shape == (NFREQ,)


def test_get_all_planes_snaps_to_the_nearest_capture(raw_dir):
    raw_dir("ants11_a.h5", {"ant": _dut()}, timestamp=1000.0)
    raw_dir("ants11_b.h5", {"ant": _dut()}, timestamp=5000.0)
    s11 = S11(raw_dir.dir).calibrate(None, None)

    assert s11.get_all_planes("ant", 1400.0)[0] == 1000.0
    assert s11.get_all_planes("ant", 4000.0)[0] == 5000.0


def test_get_all_planes_returns_a_copy(loaded):
    """Callers must not be able to reach into the object's state."""
    s11 = S11(loaded).calibrate(None, None)
    _, planes = s11.get_all_planes("ant", 1000.0)
    planes["bogus"] = None

    assert "bogus" not in s11.ant[1000.0]


def test_get_all_planes_for_an_unknown_dut_lists_what_loaded(loaded):
    s11 = S11(loaded).calibrate(None, None)
    with pytest.raises(ValueError, match="loaded DUTs are"):
        s11.get_all_planes("nope", 1000.0)


def test_get_all_duts_sweeps_the_dut_axis(loaded, tmp_path):
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))
    times, s11s = s11.get_all_duts(1000.0, "vna")

    assert set(s11s) == {"ant", "amb", "load", "noise", "rec"}
    assert set(times) == set(s11s)
    # each DUT matched its own nearest capture: rec lives in its own
    # file at a different time
    assert times["ant"] == 1000.0
    assert times["rec"] == 1100.0
    np.testing.assert_array_equal(s11s["ant"], s11.ant[1000.0]["vna"])
    np.testing.assert_array_equal(s11s["rec"], s11.rec[1100.0]["vna"])


def test_get_all_duts_skips_duts_without_that_plane(loaded, tmp_path):
    """load/noise/rec never reach the LNA plane in 2026."""
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))
    _, s11s = s11.get_all_duts(1000.0, "lna")

    assert set(s11s) == {"ant", "amb"}


def test_get_all_duts_default_plane_spans_differing_depths(loaded, tmp_path):
    """ "default" resolves per DUT, so every DUT is present."""
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))
    _, s11s = s11.get_all_duts(1000.0)

    assert set(s11s) == {"ant", "amb", "load", "noise", "rec"}
    np.testing.assert_array_equal(s11s["ant"], s11.ant[1000.0]["lna"])
    np.testing.assert_array_equal(s11s["load"], s11.load[1000.0]["vna"])


def test_get_all_duts_on_an_unreached_plane_is_empty_not_an_error(loaded):
    s11 = S11(loaded).calibrate(None, None)  # nothing past "vna"
    times, s11s = s11.get_all_duts(1000.0, "lna")

    assert times == {} and s11s == {}


def test_the_three_slices_agree_at_their_intersection(loaded, tmp_path):
    """All three accessors must return the same array for the same
    (dut, timestamp, plane) cell."""
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))

    _, by_plane = s11.get_all_planes("ant", 1000.0)
    _, by_dut = s11.get_all_duts(1000.0, "lna")
    times, by_time = s11.get_all_s11s("ant", "lna")

    np.testing.assert_array_equal(by_plane["lna"], by_dut["ant"])
    np.testing.assert_array_equal(
        by_dut["ant"], by_time[list(times).index(1000.0)]
    )


def test_plane_summaries(loaded, tmp_path):
    s11 = S11(loaded).calibrate(_sparams_npz(tmp_path), _osl_npz(tmp_path))

    assert s11.rec_planes == ["default", "dut", "raw", "vna"]
    # deepest single antenna-mode capture is ant/amb, not load/noise
    assert s11.ant_planes == ["default", "dut", "lna", "raw", "vna"]


def test_plane_summaries_are_none_without_those_duts(raw_dir):
    raw_dir("ants11_20260101_000000Z.h5", {"ant": _dut()})
    assert S11(raw_dir.dir).rec_planes is None


# -- delay domain ---------------------------------------------------


def test_delay_axis_is_ns(loaded):
    s11 = S11(loaded)
    df = FREQS[1] - FREQS[0]

    np.testing.assert_allclose(s11.dlys, np.fft.fftfreq(NFREQ, d=df) * 1e9)


def test_get_dly_shapes_and_time(loaded):
    s11 = S11(loaded).calibrate(None, None)
    time, dly = s11.get_dly("ant", 1000.0)

    assert time == 1000.0
    assert dly.shape == (NFREQ,)
    assert np.isrealobj(dly)


def test_get_all_dlys_transforms_each_capture(raw_dir):
    for i, ts in enumerate([1000.0, 2000.0]):
        raw_dir(f"ants11_{i}.h5", {"ant": _dut(0.1 * (i + 1))}, timestamp=ts)
    s11 = S11(raw_dir.dir).calibrate(None, None)
    times, dlys = s11.get_all_dlys("ant", "vna")

    assert dlys.shape == (2, NFREQ)
    _, one = s11.get_dly("ant", times[0], "vna")
    np.testing.assert_allclose(dlys[0], one)


def test_delay_transform_is_windowed(loaded):
    """A flat S11 windowed then transformed is the window's own
    transform -- not the unwindowed delta function."""
    s11 = S11(loaded)
    flat = np.ones(NFREQ, dtype=complex)

    got = s11.delay_transform(flat)
    expected = np.abs(np.fft.fft(blackmanharris(NFREQ, sym=False)))
    np.testing.assert_allclose(got, expected)
    assert not np.allclose(got, np.abs(np.fft.fft(flat)))


def test_repr_mentions_state(loaded):
    s11 = S11(loaded)
    assert "uncalibrated" in repr(s11)
    assert "calibrated" in repr(s11.calibrate(None, None))
