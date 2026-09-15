import dataclasses

import numpy as np
import pytest
from test_data import observation, reflection

from eigsep_cal.data import Observation, Reflection


def assert_same(a, b):
    assert type(a) is type(b)
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
            assert x.dtype == y.dtype, f.name
        elif f.name in getattr(type(a), "_ARRAY_DICTS", ()):
            assert x.keys() == y.keys()
            for key in x:
                np.testing.assert_array_equal(x[key], y[key])
        else:
            assert (dict(x) if hasattr(x, "keys") else x) == (
                dict(y) if hasattr(y, "keys") else y
            ), f.name


@pytest.mark.parametrize(
    "make", [observation, reflection, lambda: observation(chan=None)]
)
def test_roundtrip(tmp_path, make):
    obj = make()
    path = tmp_path / "obj.npz"
    obj.save(path)
    assert_same(obj, type(obj).load(path))


def test_roundtrip_without_npz_suffix(tmp_path):
    obj = reflection()
    obj.save(tmp_path / "run1")
    assert_same(obj, Reflection.load(tmp_path / "run1"))


def test_roundtrip_without_npz_suffix_str_paths(tmp_path):
    obj = reflection()
    obj.save(str(tmp_path / "run1"))
    assert_same(obj, Reflection.load(str(tmp_path / "run1")))


def test_empty_covariates_roundtrip(tmp_path):
    obj = observation(covariates={}, provenance={})
    obj.save(tmp_path / "o.npz")
    assert dict(Observation.load(tmp_path / "o.npz").covariates) == {}


def test_wrong_kind_raises(tmp_path):
    reflection().save(tmp_path / "r.npz")
    with pytest.raises(ValueError, match="Reflection"):
        Observation.load(tmp_path / "r.npz")


def test_wrong_spec_version_raises(tmp_path):
    reflection().save(tmp_path / "r.npz")
    with np.load(tmp_path / "r.npz") as d:
        payload = dict(d)
    payload["spec_version"] = np.array("99")
    np.savez(tmp_path / "r99.npz", **payload)
    with pytest.raises(ValueError, match="spec_version"):
        Reflection.load(tmp_path / "r99.npz")


def test_json_fields_reload_as_json_types(tmp_path):
    """Documented in io.save: tuples reload as lists, non-str dict keys
    as str. Pinned so the behaviour cannot change silently."""
    obj = observation(provenance={"files": ("a.h5", "b.h5"), 7: {"n": 16}})
    obj.save(tmp_path / "o.npz")
    got = dict(Observation.load(tmp_path / "o.npz").provenance)
    assert got == {"files": ["a.h5", "b.h5"], "7": {"n": 16}}


def test_provenance_must_be_json(tmp_path):
    obj = observation(provenance={"bad": object()})
    with pytest.raises(TypeError):
        obj.save(tmp_path / "o.npz")
