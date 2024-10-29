import gc
from typing import Union

import cv2

from ..core.typing import GetItem, Numeric
from ..libs.numpy import np
from ..core.dtype import Int, UInt8, UInt32, Float32


def shift_scale(value: Union[np.ndarray, Numeric], old_scale: GetItem, new_scale: GetItem):
    """
    Shift value from one scale to another.

    :param value: : Union[np.ndarray, Numeric];
    :param old_scale: GetItem;
    :param new_scale: GetItem;
    :return: Union[np.ndarray, Numeric];
    """
    old_sacle_min, old_scale_max = old_scale
    new_scale_min, new_scale_max = new_scale
    old_scale_range = old_scale_max - old_sacle_min
    new_scale_range = new_scale_max - new_scale_min

    return ((value - old_sacle_min) * new_scale_range / old_scale_range) + new_scale_min


def uint32_to_rgba(uint32_img: Union[UInt32, np.ndarray], width: Int, height: Int) -> Union[UInt8, np.ndarray]:
    """
    Convert image from form where each RGBA value of 8 bits each (uint8) is stored as single 32 bit
    value (uint32) back to form of (h x w x 4) (each 4 values are RGBA uint8).

    :param uint32_img: Union[UInt32, np.ndarray];
    :param width: Int;
    :param height: Int;
    :return: Union[UInt8, np.ndarray];
    """
    r = UInt8(uint32_img)
    g = UInt8(uint32_img >> 8)
    b = UInt8(uint32_img >> 16)
    a = UInt8(uint32_img >> 24)
    del uint32_img
    gc.collect()
    return np.stack([r, g, b, a]).T.reshape(height, width, 4)


def rgba_to_gray(rgba: Union[UInt8, UInt32, np.ndarray]) -> Union[UInt8, UInt32, np.ndarray]:
    # noinspection PyUnresolvedReferences
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    return gray.astype(rgba.dtype)


def expand_as_rgba(image: np.ndarray) -> np.ndarray:
    """
    Takes 3 or 4 channel images (RGB, RGBA) and add alpha layer if missing.

    :param image: np.ndarray; image of shape (height, width, channels)
    :return: np.ndarray; image of shape (height, width, 4)
    """
    # Add alpha-channels, if they are not provided
    try:
        if image.shape[2] == 3:
            return np.dstack((image, np.ones(image.shape[:2] + (1,)) * 255)).astype(np.uint8)
        elif image.shape[2] != 4:
            raise IndexError("")
    except IndexError:
        raise TypeError("Invalid image type. Expecting RGB or RGBA!")
    return image


def rgb_to_luminosity(rgb: np.ndarray) -> np.ndarray:
    """
    Compute approximate luminosity for given RGB/A image.


    :param rgb: np.ndarray; Image in shape (h, w, 3 or 4)
    :return: np.ndarray; Luminosity in shape (h, w)
    """
    int_dtypes = (UInt8, UInt32, Int)
    original_type = rgb.dtype
    if original_type in int_dtypes:
        rgb = rgb.astype(Float32) / 255.0
    luminosity = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return (luminosity * 255).astype(original_type) if original_type in int_dtypes else luminosity
