from abc import ABC
from pathlib import Path
from unittest import TestCase


class DataTestCase(TestCase, ABC):
    DATA_DIR = Path(__file__).parent / 'data'


class BrightnessContrastTestCase(DataTestCase):
    def test_input_output_dtype(self):
        pass

    def test_legacy_brightness_and_contrast(self):
        pass

    def test_brightness_and_contrast(self):
        pass

    def test_input_output_legacy_brightness_dtype(self):
        pass

    def test_input_output_legacy_contrast_dtype(self):
        pass
