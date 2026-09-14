import numpy as np
import pytest

from eigsep_cal import _validate as v


class TestArrays:
    def test_float_array_is_float64_readonly_copy(self):
        src = np.arange(3)
        arr = v.float_array("x", src)
        assert arr.dtype == np.float64
        src[0] = 99
        assert arr[0] == 0.0
        with pytest.raises(ValueError):
            arr[0] = 1.0

    def test_float_array_rejects_complex(self):
        with pytest.raises(TypeError, match="x must be real"):
            v.float_array("x", [1 + 1j])

    def test_complex_array_dtype(self):
        assert v.complex_array("g", [0.1, 0.2]).dtype == np.complex128

    def test_ndim_checked(self):
        with pytest.raises(ValueError, match="ndim"):
            v.float_array("x", [[1.0]], ndim=1)


class TestFrequencies:
    @pytest.mark.parametrize("bad", [[], [2.0, 1.0], [1.0, 1.0], [[1.0, 2.0]]])
    def test_freqs_rejects_bad_grids(self, bad):
        with pytest.raises(ValueError):
            v.freqs_mhz(bad)

    def test_same_grid(self):
        a = v.freqs_mhz([50.0, 60.0])
        v.same_grid("b", np.array([50.0, 60.0]), a)
        with pytest.raises(ValueError, match="never interpolates"):
            v.same_grid("b", np.array([50.0, 61.0]), a)


class TestShapesAndStates:
    def test_check_shape_wildcards(self):
        v.check_shape("x", np.zeros(4), (4,), (None, 4))
        v.check_shape("x", np.zeros((3, 4)), (4,), (None, 4))
        with pytest.raises(ValueError, match="x has shape"):
            v.check_shape("x", np.zeros((4, 3)), (4,), (None, 4))

    def test_states(self):
        arr = v.states("state", ["RFANT", "RFAMB"])
        assert arr.dtype.kind == "U"
        with pytest.raises(ValueError, match="VNAANT"):
            v.states("state", ["RFANT", "VNAANT"])
