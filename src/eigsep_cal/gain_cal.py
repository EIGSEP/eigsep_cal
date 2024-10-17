from eigsep_corr import io
import numpy as np

DATA_PATH = "/home/christian/Documents/research/eigsep/eigsep_cal/gain_calibration/data/oct24"

def read_dat(module, id_num, data_path=DATA_PATH):
    """
    Parameters
    ----------
    module : str
        ``fem'', ``pam'', ``snap'', or ``fiber''
    id_num : str
        The id number of the module. Labelled on the module itself.
    keys : list of str
        Which key to read from the data (e.g. polarization or input channel).

    Returns
    -------
    data : np.ndarray
        The data read from the module.

    """
    # default data type is big-endian int32
    dtype = io.build_dtype("int32", ">")
    if module == "snap":
        d = np.load(f"{DATA_PATH}/{module}/C000091/{id_num}.npz")
    else:
        d = np.load(f"{DATA_PATH}/{module}/{module}{id_num}.npz")
    data = np.frombuffer(d["3"], dtype=dtype).astype(float)
    return data


class SignalChain:

    # the inputs used for the reference signal to calibrate to
    ref_config = {
        "fem": "032_east",
        "pam": "377_north",
        "snap": "E6",
        "fiber": "G",
    }
    ref_pam_atten = 8

    # the signal to calibrate to
    ref_snap = ref_config["snap"]
    ref_signal = read_dat("snap", ref_snap)

    def __init__(self):
        self.get_ref_gains()
        self.modules = {}
        self.gain_ratios = {}

    def get_ref_gains(self):
        self.ref_gains = {}
        for module, id_num in self.ref_config.items():
            data = read_dat(module, id_num)
            self.ref_gains[module] = data

    def add_module(self, module, id_num):
        self.modules[module] = id_num
        data = read_dat(module, id_num)
        r = data / self.ref_gains[module]
        self.gain_ratios[module] = r

    @property
    def ex_bandpass(self):
        """
        The expected bandpass of the signal chain
        """
        sig = self.ref_signal
        for module in self.modules:
            sig = sig * self.gain_ratios[module]
        return sig
