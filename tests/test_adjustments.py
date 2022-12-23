from numpy import array_equal
from parameterized import parameterized

from photoshop import brightness_contrast
from photoshop.core.dtype import UInt8

from tests.utils import DataTestCase


class BrightnessContrastTestCase(DataTestCase):
    @parameterized.expand([(True, ), (False, )])
    def test_io_original_unchanged(self, legacy):
        rgba = self.EXPECTED_RGBA.copy()
        _ = brightness_contrast(rgba, brightness=10, contrast=10, use_legacy=legacy)
        assert array_equal(rgba, self.EXPECTED_RGBA) is True

    @parameterized.expand([(True,), (False,)])
    def test_io_dtype(self, legacy):
        rgba = self.EXPECTED_RGBA
        adjusted = brightness_contrast(rgba, brightness=10, contrast=10, use_legacy=legacy)
        assert adjusted.dtype == UInt8

    def test_brightness_and_contrast(self):
        pass
