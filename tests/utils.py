import cv2

from abc import ABC
from pathlib import Path
from unittest import TestCase


class DataTestCase(TestCase, ABC):
    DATA_DIR = Path(__file__).parent / 'data'
    RGBA_IMAGE_PATH = DATA_DIR / 'RGBA_comp.png'

    # noinspection PyMethodParameters
    def setUp(cls) -> None:
        super(DataTestCase, cls).setUp()
        cls.EXPECTED_RGBA = cls.load_unchanged_as_rgba()

    @classmethod
    def load_unchanged_as_rgba(cls):
        # noinspection PyUnresolvedReferences
        bgra = cv2.imread(str(cls.RGBA_IMAGE_PATH), cv2.IMREAD_UNCHANGED)
        # noinspection PyUnresolvedReferences
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
