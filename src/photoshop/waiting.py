from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from photoshop.blend.methods import normal
from photoshop.core.dtype import Float32
from photoshop.filters.blur.gaussian import gaussian_blur
from photoshop.ops.transform import expand_as_rgba


def drop_shadow(
        rgba: np.ndarray,
        blend_mode: str = 'normal',
        opacity: float = 1.0,
        angle: float = 120,
        distance: int = 10,
        size: int = 5,
        color: Tuple = (0, 0, 0)
):
    # Always work with RGBA image
    rgba = expand_as_rgba(rgba)

    # Compute shift based on light direction.
    dx = -int(distance * np.cos(np.radians(angle)))
    dy = int(distance * np.sin(np.radians(angle)))

    # Create shadow image based on supplied color
    shadow = np.zeros(rgba.shape)
    shadow[..., :3] = color
    shadow[..., 3] = rgba[..., 3] * opacity

    # Translate shadow to desired coordinates.
    height, width = shadow.shape[:2]

    # Define the transformation matrix
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=Float32)

    # Apply the affine transformation to move the image
    # noinspection PyUnresolvedReferences
    shadow = cv2.warpAffine(shadow, matrix, (width, height))

    # Apply Gaussian blur upon shadow
    shadow = gaussian_blur(shadow, radius=size)

    # Create a composite image with preserved alpha
    img_out = normal(rgba.copy(), shadow)

    plt.imshow(img_out)
    plt.show()
