"""Conventions fixed by the API spec (docs/api.md)."""

import numpy as np

SPEC_VERSION = "0"

#: Switch states that may reach a stage (spec § 3).
STATES = (
    "RFANT",
    "RFAMB",
    "RFNON",
    "RFNOFF",
    "RFSP1_SHORT",
    "RFSP1_OPEN",
)

#: Antennas (spec § 5.1).
ANTENNAS = ("box-air", "box-gnd")

#: D5 correlator channel spacing in MHz: f_k = k * 250 / 1024 (spec § 2).
D5_CHANNEL_MHZ = 250 / 1024

#: Default equivalent noise bandwidth in Hz: one channel spacing, until
#: the SNAP filterbank's ENBW is measured (spec § 4.4).
D5_ENBW_HZ = D5_CHANNEL_MHZ * 1e6


def d5_freqs_mhz(chan):
    """Frequencies in MHz of D5 correlator channels (integers 0-1023)."""
    chan = np.asarray(chan)
    if not np.issubdtype(chan.dtype, np.integer):
        raise ValueError("chan must be integers")
    if np.any((chan < 0) | (chan > 1023)):
        raise ValueError("chan must lie in 0..1023")
    return chan * D5_CHANNEL_MHZ
