"""Save and load eigsep_cal objects as npz (spec § 5.6)."""

import dataclasses
import json

import numpy as np

from .conventions import SPEC_VERSION

_DICT = "__dict__."
_JSON = "__json__."


def save(obj, path):
    """Write a dataclass object to *path* as npz with ``spec_version``."""
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
    np.savez(path, **payload)


def load(path, cls):
    """Read an object of class *cls* written by :func:`save`."""
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
