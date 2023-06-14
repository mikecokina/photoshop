import cv2
from .libs.numpy import np


# noinspection PyUnresolvedReferences
def load_rgba(path: str) -> np.ndarray:
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
