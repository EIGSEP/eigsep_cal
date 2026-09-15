"""Save and load eigsep_cal objects as npz (spec § 5.6)."""

import dataclasses
import json
from pathlib import Path

import numpy as np

from .conventions import SPEC_VERSION

_DICT = "__dict__."
_JSON = "__json__."


def _npz_path(path):
    """*path* as a :class:`~pathlib.Path`, with ``.npz`` appended if
    it is missing (``np.savez`` appends it; :func:`np.load` does not)."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    return path


def save(obj, path):
    """Write a dataclass object to *path* as npz with ``spec_version``.

    Arrays are stored as they are, and each dict named in the class's
    ``_ARRAY_DICTS`` (``Observation.covariates``) as one array per key.
    Every other field, including dicts such as ``provenance`` or
    ``meta``, is stored as JSON, so it does not round-trip exactly:
    tuples reload as lists and non-string dict keys as strings. Values
    JSON cannot encode, such as numpy arrays, raise ``TypeError``.

    Parameters
    ----------
    obj : dataclass
        The object to write, for example an ``Observation``.
    path : str or path-like
        Destination; ``.npz`` is appended if it is missing.
    """
    array_dicts = getattr(type(obj), "_ARRAY_DICTS", ())
    payload = {
        "spec_version": np.array(SPEC_VERSION),
        "kind": np.array(type(obj).__name__),
    }
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            payload[f.name] = value
        elif f.name in array_dicts:
            payload[_DICT + f.name] = np.array(json.dumps(list(value)))
            for key, arr in value.items():
                payload[f"{_DICT}{f.name}.{key}"] = arr
        else:
            if hasattr(value, "keys"):
                value = dict(value)
            payload[_JSON + f.name] = np.array(json.dumps(value))
    np.savez(_npz_path(path), **payload)


def load(path, cls):
    """Read an object of class *cls* written by :func:`save`.

    JSON-stored fields come back as JSON types (see :func:`save`): a
    tuple in ``provenance`` reloads as a list, and a non-string dict
    key as a string.

    Parameters
    ----------
    path : str or path-like
        A file written by :func:`save`; ``.npz`` is appended if it is
        missing.
    cls : type
        The class to build; it must match the file's ``kind``.

    Returns
    -------
    obj : cls
    """
    path = _npz_path(path)
    with np.load(path, allow_pickle=False) as d:
        version = str(d["spec_version"])
        if version != SPEC_VERSION:
            raise ValueError(
                f"{path} has spec_version {version}; "
                f"this eigsep_cal reads {SPEC_VERSION}"
            )
        kind = str(d["kind"])
        if kind != cls.__name__:
            raise ValueError(f"{path} holds a {kind}, not a {cls.__name__}")
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in d.files:
                kwargs[f.name] = d[f.name]
            elif _DICT + f.name in d.files:
                keys = json.loads(str(d[_DICT + f.name]))
                kwargs[f.name] = {
                    key: d[f"{_DICT}{f.name}.{key}"] for key in keys
                }
            elif _JSON + f.name in d.files:
                kwargs[f.name] = json.loads(str(d[_JSON + f.name]))
    return cls(**kwargs)
