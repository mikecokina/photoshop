import numpy as np
from numpy import array_equal
from numpy.testing import assert_array_equal
from parameterized import parameterized

from photoshop import brightness_contrast, auto_contrast
from photoshop.core.dtype import UInt8

from tests.utils import DataTestCase, read_image_from_json


class BrightnessContrastTestCase(DataTestCase):
    @parameterized.expand([(True, ), (False, )])
    def test_brightness_and_contrast_io_original_unchanged(self, legacy):
        rgba = self.DEFAULT_RGBA.copy()
        _ = brightness_contrast(rgba, brightness=10, contrast=10, use_legacy=legacy)
        assert array_equal(rgba, self.DEFAULT_RGBA) is True

    @parameterized.expand([(True,), (False,)])
    def test_brightness_and_contrast_io_dtype(self, legacy):
        rgba = self.DEFAULT_RGBA
        adjusted = brightness_contrast(rgba, brightness=10, contrast=10, use_legacy=legacy)
        assert adjusted.dtype == UInt8

    @parameterized.expand([(True,), (False,)])
    def test_brightness_and_contrast(self, legacy):
        brit, cntr = -20, 20
        rgba = self.DEFAULT_RGBA
        adjusted = brightness_contrast(rgba, brightness=brit, contrast=cntr, use_legacy=legacy)
        path = self.DATA_DIR / f'RGBA_m20brit_p20cntr_{int(legacy)}.npy'
        expected = np.load(path)
        assert_array_equal(adjusted, expected)

    def test_auto_contrast_unchanged(self):
        rgba = self.DEFAULT_RGBA.copy()
        _ = auto_contrast(rgba)
        assert array_equal(rgba, self.DEFAULT_RGBA) is True

    def test_autocontrast_io_dtype(self):
        rgba = self.DEFAULT_RGBA
        adjusted = auto_contrast(rgba)
        assert adjusted.dtype == UInt8

    def test_autocontrast(self):
        rgba = self.DEFAULT_RGBA
        adjusted = auto_contrast(rgba)

        path = self.DATA_DIR / f'RGBA_autocontrast.npy'
        expected = np.load(path)
        assert_array_equal(adjusted, expected)
