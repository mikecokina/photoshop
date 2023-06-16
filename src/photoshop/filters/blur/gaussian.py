import cv2

from ...core.dtype import Int
from ...libs.numpy import np


def gaussian_blur(rgba: np.ndarray, radius: Int) -> np.ndarray:
    """
    Equations to transform pixels radius from Photoshop to CV2 parameters is derived empirically.

    PIL equivalent:

        def apply_blur(image, radius):
            blurred_image = image.filter(ImageFilter.BLUR)
            return blurred_image

    :param rgba: np.ndarray; Image in shape (h x w x ch)
    :param radius: Int;
    :return: np.ndarray;
    """
    original_type = rgba.dtype

    sigma = (radius - 0.5) / 3.0
    kernel_size = int(2 * round(2 * sigma) + 1)
    # noinspection PyUnresolvedReferences
    blurred_image = cv2.GaussianBlur(rgba, (kernel_size, kernel_size), sigma)
    return blurred_image.astype(original_type)
