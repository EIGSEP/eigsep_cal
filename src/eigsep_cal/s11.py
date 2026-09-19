"""Field S11 measurements: raw captures in, calibrated planes out.

Load with ``S11(path, year)``, then calibrate::

    raw --[de-embed VNA internal OSL]--> vna
        --[de-embed VNA-leg path]------> dut
        --[embed RF/LNA-leg path]------> lna

See the README for the calibration chain, the per-season switch
dictionaries and the three accessors.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.signal.windows import blackmanharris

from . import calkit

# +
# Which switch path(s) must be de-embedded (VNA leg) / embedded (RF or
# LNA leg) to get from the VNA's internal-OSL reference plane to each
# DUT's physical location and on to the LNA input. An empty list means
# that DUT sits at the previous plane already -- nothing to de-embed
# or embed, so the chain stops there.
DEEMBED_DICT26 = {
    "amb": ["VNAAMB"],
    "ant": ["VNAANT"],
    "load": [],
    "noise": [],
    "sp1_open": ["VNASP1"],
    "sp1_short": ["VNASP1"],
    "sp1": ["VNASP1"],
    "rec": ["VNARF"],
}

EMBED_DICT26 = {
    "amb": ["RFAMB"],
    "ant": ["RFANT"],
    "load": [],
    "noise": [],
    "sp1_open": ["RFSP1"],
    "sp1_short": ["RFSP1"],
    "sp1": ["RFSP1"],
    "rec": [],
}

DEEMBED_DICT25 = {
    "ant": ["VNAANT"],
    "load": ["VNAN"],
    "noise": ["VNAN"],
    "rec": ["VNARF"],
}

EMBED_DICT25 = {
    "ant": ["RFANT"],
    "load": ["RFN"],
    "noise": ["RFN"],
    "rec": [],
}

# ``year=None`` has no switch S-parameters: nothing to de-embed or
# embed, so every capture stops at the "vna" plane.
DEEMBED_DICTS = {None: {}, 2025: DEEMBED_DICT25, 2026: DEEMBED_DICT26}
EMBED_DICTS = {None: {}, 2025: EMBED_DICT25, 2026: EMBED_DICT26}
# -

#: Characterization files shipped with the package. See data/README.md.
DATA_DIR = Path(__file__).parent / "data"


def _packaged_for(year):
    return {
        "switchpaths": DATA_DIR / f"switch_sparams{year}.npz",
        "osldata": DATA_DIR / f"calibrated_internal_osl{year}.npz",
    }


#: Season -> its shipped files. A season with no files yet still
#: appears here, so the error names the path to drop them at.
PACKAGED_FILES = {year: _packaged_for(year) for year in (2025, 2026)}

#: Sentinel: use :data:`PACKAGED_FILES` for this object's season.
PACKAGED = "packaged"

#: Calibration planes, ordered shallowest to deepest.
PLANES = ("raw", "vna", "dut", "lna")

# 2025-season files carry no "metadata_snapshot_unix" in their header:
# the capture time lives only in the filename, e.g.
# "ants11_20250719_102527.h5" -> "20250719_102527", read as local
# Mountain Standard Time (a fixed UTC-7, not the DST-shifting zone).
_MST = timezone(timedelta(hours=-7))
_FNAME_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def timestamp_from_filename(path):
    """Unix timestamp from a 2025 filename, read as MST.

    ``ants11_20250719_102527.h5`` -> 2025-07-19 10:25:27.
    """
    match = _FNAME_TIMESTAMP_RE.search(Path(path).stem)
    if not match:
        raise ValueError(
            f"{Path(path).name}: no YYYYMMDD_HHMMSS timestamp in the "
            "filename (expected for 2025-season data)"
        )
    dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return dt.replace(tzinfo=_MST).timestamp()


class S11Sample:
    """A (DUT, receiver) pair from :meth:`S11.get_s11`.

    The two times are the captures matched, which can differ.
    """

    __slots__ = ("dut", "dut_time", "dut_s11", "rec_time", "rec_s11")

    def __init__(self, dut, dut_time, dut_s11, rec_time, rec_s11):
        self.dut = dut
        self.dut_time = dut_time
        self.dut_s11 = dut_s11
        self.rec_time = rec_time
        self.rec_s11 = rec_s11

    def __repr__(self):
        return (
            f"S11Sample(dut={self.dut!r}, dut_time={self.dut_time!r}, "
            f"rec_time={self.rec_time!r})"
        )


class S11:
    """Raw field S11 captures and the planes derived from them.

    Construction only reads files; call :meth:`calibrate` for the
    ``vna``/``dut``/``lna`` planes.

    Parameters
    ----------
    path : str or Path
        A directory of raw S11 HDF5 files, or a single file.
    year : int or None, optional
        2025 or 2026 (default). Picks the switch dictionaries; 2025
        also takes timestamps from filenames. ``None`` has no switch
        S-parameters, so only the ``"raw"`` and ``"vna"`` planes are
        produced, against an ideal OSL model; timestamps then come
        from whichever source a file has.
    pattern : str, optional
        Glob used when ``path`` is a directory.

    Attributes
    ----------
    freqs, freqs_mhz, dlys : np.ndarray
        Frequency axis (Hz, MHz) and its conjugate delay axis (ns).
    duts : list of str
        DUTs loaded; each is also an attribute, e.g. ``self.ant`` --
        ``{timestamp: {plane: s11}}``.
    calibrated : bool
    skipped : list of tuple
        ``(filename, reason)`` for files left out at load time.
    """

    def __init__(self, path, year=2026, pattern="*.h5"):
        if year not in DEEMBED_DICTS:
            allowed = sorted(y for y in DEEMBED_DICTS if y is not None)
            raise ValueError(
                f"year must be None or one of {allowed}, got {year!r}"
            )
        self.path = Path(path)
        self.year = year
        self.calibrated = False
        self.skipped = []

        paths = self._resolve_paths(self.path, pattern)
        loaded = [self._read_one(p) for p in paths]
        self.freqs = self._adopt_grid(loaded)
        self._ingest(loaded)

    # -- loading ----------------------------------------------------

    @staticmethod
    def _resolve_paths(path, pattern):
        """Every raw file to read, for a file or a directory."""
        if path.is_dir():
            paths = sorted(path.glob(pattern))
            if not paths:
                raise ValueError(f"no files matching {pattern!r} in {path}")
            return paths
        if path.is_file():
            return [path]
        raise ValueError(f"no such file or directory: {path}")

    def _read_one(self, path):
        """Read one raw file. eigsep_base is imported here, not at
        module scope: it is not on PyPI, so only this path needs it.
        """
        try:
            from eigsep_base import io
        except ImportError as e:
            raise ImportError(
                "reading raw S11 files needs eigsep_base, which is not "
                "installed. Install it from a sibling checkout with "
                "'pip install -e ../eigsep_base'. The rest of eigsep_cal "
                "does not require it."
            ) from e

        data, cal_data, hdr, meta = io.read_s11_file(path)
        timestamp = self._timestamp(path, hdr)
        return {
            "path": path,
            "data": data,
            "cal_data": cal_data,
            "hdr": hdr,
            "timestamp": timestamp,
            "freqs": np.asarray(hdr["freqs"], dtype=float),
        }

    def _timestamp(self, path, hdr):
        """Capture time: the header's, or the filename's for 2025.

        ``year=None`` takes whichever is there, so a quick look works
        on either season's files.
        """
        if self.year == 2025:
            return timestamp_from_filename(path)
        if self.year is None and "metadata_snapshot_unix" not in hdr:
            return timestamp_from_filename(path)
        return hdr["metadata_snapshot_unix"]

    def _adopt_grid(self, loaded):
        """Pick the frequency axis shared by most files.

        Directories can mix sweep spans. Nothing interpolates, so the
        majority axis wins and the rest go to ``skipped``.
        """
        if not loaded:
            raise ValueError(f"no readable S11 files in {self.path}")
        groups = {}
        for entry in loaded:
            groups.setdefault(entry["freqs"].tobytes(), []).append(entry)
        best = max(groups.values(), key=len)
        for entries in groups.values():
            if entries is best:
                continue
            for entry in entries:
                span = entry["freqs"][-1] / 1e6
                self.skipped.append(
                    (
                        entry["path"].name,
                        f"{entry['freqs'].size} points out to {span:g} MHz, "
                        f"not the {best[0]['freqs'].size}-point axis used "
                        "by most files here",
                    )
                )
        return best[0]["freqs"]

    def _ingest(self, loaded):
        """Sort raw captures and internal-OSL sets into the object."""
        self.duts = []
        self._planes = {}
        self.osls = {"ant": {}, "rec": {}}
        for entry in loaded:
            if not np.array_equal(entry["freqs"], self.freqs):
                continue  # already recorded in self.skipped
            timestamp = entry["timestamp"]
            for dut, s11 in entry["data"].items():
                s11 = np.asarray(s11)
                if np.any(s11 == 0):
                    continue  # unmeasured/invalid capture
                if dut not in self._planes:
                    self._planes[dut] = {}
                    self.duts.append(dut)
                    setattr(self, dut, self._planes[dut])
                self._planes[dut][timestamp] = {"raw": s11}
            osl = self._internal_osl(entry)
            if osl is not None:
                mode = entry["hdr"].get("mode")
                if mode in self.osls:
                    self.osls[mode][timestamp] = osl
        self.duts.sort()

    def _internal_osl(self, entry):
        """The file's own VNAO/VNAS/VNAL set, or None."""
        cal_data = entry["cal_data"]
        try:
            osl = np.array(
                [cal_data["VNAO"], cal_data["VNAS"], cal_data["VNAL"]]
            )
        except KeyError as e:
            # Per-file, unlike the whole-run inputs to calibrate(): a
            # file with no internal-OSL capture is skipped, it does
            # not abort every other file in the directory.
            self.skipped.append(
                (entry["path"].name, f"no internal-OSL data ({e})")
            )
            return None
        if np.any(osl == 0):
            return None  # unmeasured/invalid internal OSL set
        return osl

    # -- calibration ------------------------------------------------

    def calibrate(self, switchpaths=PACKAGED, osldata=PACKAGED):
        """Walk every capture down the calibration chain.

        Defaults to this season's packaged files. ``None`` for both
        gives the quick-look path: an ideal (+1, -1, 0) OSL model
        reaching ``"vna"`` only.

        Parameters
        ----------
        switchpaths : str or Path or None, optional
            npz of switch-path S-parameters, keyed by state name.
        osldata : str or Path or None, optional
            npz of characterized OSL standards ``O``/``S``/``L``.

        Returns
        -------
        S11
            ``self``, so calls can be chained.
        """
        switchpaths, osldata = self._resolve_characterizations(
            switchpaths, osldata
        )
        quicklook = switchpaths is None
        osl_model = self._osl_model(osldata)
        sparams = {} if quicklook else self._load_npz(switchpaths)
        deembed = DEEMBED_DICTS[self.year]
        embed = EMBED_DICTS[self.year]

        for dut, by_time in self._planes.items():
            bank = self.osls["rec" if dut == "rec" else "ant"]
            if not bank:
                raise ValueError(
                    "no usable internal-OSL captures for mode "
                    f"{'rec' if dut == 'rec' else 'ant'!r} -- cannot "
                    f"calibrate {dut!r}"
                )
            times = np.array(list(bank))
            for timestamp, planes in by_time.items():
                osl = bank[times[np.argmin(np.abs(times - timestamp))]]
                vna_sparams = calkit.network_sparams(osl_model, osl)
                planes["vna"] = calkit.de_embed_sparams(
                    vna_sparams, planes["raw"]
                )
                if not quicklook:
                    self._switch_planes(
                        planes, dut, sparams, deembed, embed, switchpaths
                    )
                planes["default"] = planes[self._deepest(planes)]
        self.calibrated = True
        return self

    def _resolve_characterizations(self, switchpaths, osldata):
        """Resolve the PACKAGED sentinel and validate the pair.

        Mixing sentinel with explicit, or giving only one, is an
        error: no plane below ``"vna"`` is reachable without both.
        """
        given = {"switchpaths": switchpaths, "osldata": osldata}
        if self.year is None:
            if switchpaths in (PACKAGED, None) and osldata in (
                PACKAGED,
                None,
            ):
                return None, None
            raise ValueError(
                "there are no switch S-parameters for year=None, so "
                "only the 'raw' and 'vna' planes are available. Pass "
                "year=2025 or year=2026 to use characterization files."
            )
        if (switchpaths is PACKAGED) != (osldata is PACKAGED):
            raise ValueError(
                "switchpaths and osldata must both default to the "
                "packaged files or both be given explicitly; got "
                f"{ {k: v for k, v in given.items()} }"
            )
        if switchpaths is PACKAGED:
            return self._packaged_paths()
        if (switchpaths is None) != (osldata is None):
            raise ValueError(
                "pass both switchpaths and osldata, or None for both "
                "to get the quick-look (ideal-OSL, vna-plane-only) path"
            )
        return switchpaths, osldata

    def _packaged_paths(self):
        """This season's shipped characterization files."""
        try:
            paths = PACKAGED_FILES[self.year]
        except KeyError:
            raise ValueError(
                f"no characterization files are packaged for "
                f"{self.year}; pass switchpaths and osldata explicitly, "
                "or None for both to use the quick-look path"
            ) from None
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "packaged characterization files are not installed: "
                + ", ".join(missing)
                + f". See {DATA_DIR / 'README.md'} for the expected "
                "contents, or pass the files explicitly."
            )
        return paths["switchpaths"], paths["osldata"]

    def _switch_planes(self, planes, dut, sparams, deembed, embed, fname):
        """De-embed to the DUT plane, then embed on to the LNA plane.

        A DUT missing from the season's dictionary stops the chain
        rather than raising.
        """
        names = deembed.get(dut, [])
        if names:
            planes["dut"] = calkit.de_embed_sparams(
                sparams=self._sparams_for(sparams, names[0], fname),
                gamma_prime=planes["vna"],
            )
        names = embed.get(dut, [])
        if names and "dut" in planes:
            planes["lna"] = calkit.embed_sparams(
                sparams=self._sparams_for(sparams, names[0], fname),
                gamma=planes["dut"],
            )

    @staticmethod
    def _sparams_for(sparams, name, fname):
        try:
            return sparams[name]
        except KeyError:
            raise ValueError(
                f"switch path {name!r} is not in {Path(fname).name}; "
                f"it has {sorted(sparams)}"
            ) from None

    @staticmethod
    def _deepest(planes):
        """The deepest plane this capture reached."""
        reached = [p for p in PLANES if p in planes]
        return reached[-1]

    def _osl_model(self, osldata):
        """The "true" OSL gammas, characterized or ideal."""
        if osldata is None:
            ideal = np.array([1.0, -1.0, 0.0])[:, np.newaxis]
            return np.repeat(ideal, self.freqs.size, axis=1)
        model = self._load_npz(osldata)
        model.pop("freqs", None)
        try:
            osl_model = np.array([model["O"], model["S"], model["L"]])
        except KeyError as e:
            raise ValueError(f"OSL model data missing key: {e}") from None
        if osl_model.shape[-1] != self.freqs.size:
            raise ValueError(
                f"{Path(osldata).name} is on a {osl_model.shape[-1]}-point "
                f"frequency axis but this data is on a {self.freqs.size}"
                "-point one; eigsep_cal never interpolates in frequency"
            )
        return osl_model

    @staticmethod
    def _load_npz(path):
        return dict(np.load(path))

    # -- lookups ----------------------------------------------------

    @property
    def timestamps(self):
        """Every capture time loaded, across all DUTs, sorted."""
        return sorted({t for d in self._planes.values() for t in d})

    @property
    def rec_planes(self):
        """Planes reached by the receiver-mode measurement."""
        return self._planes_for(["rec"])

    @property
    def ant_planes(self):
        """Planes reached by the antenna-mode measurements.

        The deepest single DUT/timestamp, *not* a union across DUTs.
        """
        return self._planes_for([d for d in self.duts if d != "rec"])

    def _planes_for(self, duts):
        best = None
        for dut in duts:
            for planes in self._planes.get(dut, {}).values():
                names = sorted(planes)
                if best is None or len(names) > len(best):
                    best = names
        return best

    def _by_time(self, dut):
        try:
            by_time = self._planes[dut]
        except KeyError:
            raise ValueError(
                f"no data loaded for dut={dut!r}; loaded DUTs are {self.duts}"
            ) from None
        if not by_time:
            raise ValueError(f"{dut!r} has no timestamps recorded")
        return by_time

    def _nearest_time(self, dut, timestamp):
        """The capture time for *dut* closest to *timestamp*."""
        by_time = self._by_time(dut)
        times = np.array(list(by_time))
        return float(times[np.argmin(np.abs(times - timestamp))])

    def _nearest(self, dut, timestamp, plane):
        nearest = self._nearest_time(dut, timestamp)
        planes = self._planes[dut][nearest]
        if plane not in planes:
            raise KeyError(
                f"{dut!r} at t={nearest} has no {plane!r} plane "
                f"(has: {sorted(planes)})"
            )
        return nearest, planes[plane]

    def get_s11(self, dut, timestamp, plane="default"):
        """The S11 for *dut* and the receiver's, both nearest
        *timestamp*, as an :class:`S11Sample`."""
        dut_time, dut_s11 = self._nearest(dut, timestamp, plane)
        rec_time, rec_s11 = self._nearest("rec", timestamp, plane)
        return S11Sample(dut, dut_time, dut_s11, rec_time, rec_s11)

    def get_all_planes(self, dut, timestamp):
        """Every plane one DUT reached, at one capture.

        Returns ``(time, {plane: s11})``. Includes the ``"default"``
        alias, so the deepest array appears twice.
        """
        nearest = self._nearest_time(dut, timestamp)
        return nearest, dict(self._planes[dut][nearest])

    def get_all_duts(self, timestamp, plane="default"):
        """Every DUT's S11 at one plane, at one capture.

        Returns ``({dut: time}, {dut: s11})``. DUTs are not measured
        in lockstep, so each matches its own nearest capture; one that
        never reached *plane* is left out rather than raising.
        """
        times, s11s = {}, {}
        for dut in self.duts:
            if not self._planes[dut]:
                continue
            nearest = self._nearest_time(dut, timestamp)
            planes = self._planes[dut][nearest]
            if plane in planes:
                times[dut] = nearest
                s11s[dut] = planes[plane]
        return times, s11s

    def get_all_s11s(self, dut, plane="default"):
        """Every S11 for *dut* at *plane*, in time order.

        Returns ``(times, s11s)``, the latter ``(n_times, n_freq)``.
        Captures that never reached *plane* are left out.
        """
        by_time = self._by_time(dut)
        times = sorted(t for t in by_time if plane in by_time[t])
        return (
            np.array(times),
            np.array([by_time[t][plane] for t in times]),
        )

    # -- delay domain -----------------------------------------------

    @property
    def freqs_mhz(self):
        return self.freqs / 1e6

    @property
    def dlys(self):
        """Delay axis in ns, conjugate to :attr:`freqs`."""
        df = self.freqs[1] - self.freqs[0]
        return np.fft.fftfreq(self.freqs.size, d=df) * 1e9

    def delay_transform(self, s11):
        """Blackman-Harris-windowed delay spectrum of *s11*.

        Windowing stops one delay leaking across the axis. The window
        is periodic: these are spectrum samples, not a filter.
        """
        window = blackmanharris(self.freqs.size, sym=False)
        return np.abs(np.fft.fft(np.asarray(s11) * window))

    def get_dly(self, dut, timestamp, plane="default"):
        """Delay spectrum of one capture, nearest *timestamp*.

        Returns ``(time, dly)``, ``dly`` on :attr:`dlys`.
        """
        time, s11 = self._nearest(dut, timestamp, plane)
        return time, self.delay_transform(s11)

    def get_all_dlys(self, dut, plane="default"):
        """Delay spectra of every capture for *dut*, in time order.

        Returns ``(times, dlys)``, ``dlys`` on :attr:`dlys`.
        """
        times, s11s = self.get_all_s11s(dut, plane)
        return times, self.delay_transform(s11s)

    def __repr__(self):
        state = "calibrated" if self.calibrated else "uncalibrated"
        return (
            f"S11({self.path.name!r}, year={self.year}, {state}, "
            f"duts={self.duts}, n_times={len(self.timestamps)})"
        )
