import numpy as np
from numpy import array_equal

# noinspection PyProtectedMember
from photoshop._io import load_rgba
from photoshop.core.dtype import UInt8
from tests.utils import DataTestCase


class LoadRGBATestCase(DataTestCase):
    def test_loaded(self):
        obtained = load_rgba(str(self.RGBA_IMAGE_PATH))
        array_equal(self.EXPECTED_RGBA, obtained)

    def test_loaded_pixel_range(self):
        obtained = load_rgba(str(self.RGBA_IMAGE_PATH))
        assert np.max(obtained) == 255
        assert np.min(obtained) == 0

    def test_loaded_dtypes(self):
        obtained = load_rgba(str(self.RGBA_IMAGE_PATH))
        assert obtained.dtype == UInt8
