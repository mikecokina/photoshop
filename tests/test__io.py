from abc import ABC
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np
from numpy import array_equal

# noinspection PyProtectedMember
from photoshop._io import load_rgba
from photoshop.core.dtype import UInt8


class DataTestCase(TestCase, ABC):
    DATA_DIR = Path(__file__).parent / 'data'
    RGBA_IMAGE_PATH = DATA_DIR / 'RGBA_comp.png'

    @classmethod
    def load_unchanged_as_rgba(cls):
        # noinspection PyUnresolvedReferences
        bgra = cv2.imread(str(cls.RGBA_IMAGE_PATH), cv2.IMREAD_UNCHANGED)
        # noinspection PyUnresolvedReferences
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


class LoadRGBATestCase(DataTestCase):
    # noinspection PyMethodParameters
    def setUp(cls) -> None:
        cls.EXPECTED_RGBA = cls.load_unchanged_as_rgba()

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
