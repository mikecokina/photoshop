from typing import Tuple
import numpy as np

from photoshop.core.dtype import Int, UInt8
from photoshop.ops.transform import rgb_to_luminosity, expand_as_rgba


def normal_blend_if(
        this_layer: np.ndarray,
        underlying_layer: np.ndarray,
        underlying_layer_shadows_threshold: Int = 0,
        underlying_layer_highlights_threshold: Int = 255,
) -> np.ndarray:
    """
    Initial implementation of normal conditionakl blending. Simple condition (without split of sliders in PS).

    :param this_layer: np.ndarray; Image of shape (h, w, ch)
    :param underlying_layer: np.ndarray; Image of shape (h, w, ch)
    :param underlying_layer_shadows_threshold: int;
    :param underlying_layer_highlights_threshold: int;
    :return: np.ndarray; Image of shape (h, w, 4)
    """
    # Helepr function
    def _normal_blend(fg_rgb_: np.ndarray, fg_a_: np.ndarray, bg_rgb_: np.ndarray, bg_a_: np.ndarray) -> np.ndarray:
        return (fg_rgb_ * fg_a_ + bg_rgb_ * bg_a_ * (1 - fg_a_)) / (fg_a_ + bg_a_ * (1 - fg_a_))

    # Shorten variable name
    bg_shadows_t, bg_highlights_t = underlying_layer_shadows_threshold, underlying_layer_highlights_threshold

    # Validation
    if bg_shadows_t > bg_highlights_t:
        raise ValueError(
            "Not support parameters combination. "
            "Required underlying_layer_shadows_threshold <= underlying_layer_highlights_threshold"
        )

    # Expand with alpha if missing
    foreground_array = expand_as_rgba(this_layer) / 255.0
    background_array = expand_as_rgba(underlying_layer) / 255.0

    # Calculate the luminosity of the background image
    background_luminosity = (rgb_to_luminosity(background_array) * 255.0).astype(UInt8)

    # Create the blend if condition based on the luminosity range
    condition = (background_luminosity >= bg_shadows_t) & (background_luminosity <= bg_highlights_t)
    blend_if = np.expand_dims(condition, 2)

    # Expand dimensions for further use
    foreground_rgb, background_rgb = foreground_array[:, :, :3], background_array[:, :, :3]
    foreground_a = np.expand_dims(foreground_array[:, :, 3], 2)
    background_a = np.expand_dims(background_array[:, :, 3], 2)

    # Conditional blending
    args_ = (foreground_rgb, foreground_a, background_rgb, background_a)
    blended_rgb = np.where(blend_if, _normal_blend(*args_), background_rgb)
    blended_a = np.where(blend_if, foreground_a + background_a * (1 - foreground_a), background_a)

    # Combine the blended channels back into a single RGBA image
    blended_rgba = np.dstack((blended_rgb, blended_a))
    # Scale the RGBA values back to the range [0, 255]
    blended_rgba = (blended_rgba * 255).astype(np.uint8)

    return blended_rgba


def normal_complex_blend_if(
        this_layer: np.ndarray,
        underlying_layer: np.ndarray,
        underlying_layer_shadows_range: Tuple = (0, 0),
        underlying_layer_highlights_range: Tuple = (255, 255),
) -> np.ndarray:
    """
    Initial implementation of normal conditionakl blending.

    :param this_layer: np.ndarray; Image of shape (h, w, ch)
    :param underlying_layer: np.ndarray; Image of shape (h, w, ch)
    :param underlying_layer_shadows_range: Tuple[int, int];
    :param underlying_layer_highlights_range: Tuple[int, int];
    :return: np.ndarray; Image of shape (h, w, 4)
    """
    bg_shadows, bg_highlights = underlying_layer_shadows_range, underlying_layer_highlights_range

    # Input validation.
    if bg_shadows[0] > bg_shadows[1]:
        bg_shadows = (bg_shadows[0], bg_shadows[0])

    if bg_highlights[0] > bg_highlights[1]:
        bg_highlights = (bg_highlights[1], bg_highlights[1])

    # Fallback to simple version
    if (bg_shadows[0] == bg_shadows[1]) & (bg_highlights[0] == bg_highlights[1]):
        return normal_blend_if(this_layer, underlying_layer, bg_shadows[0], bg_highlights[0])

    # Expand with alpha if missing
    foreground_array = expand_as_rgba(this_layer) / 255.0
    background_array = expand_as_rgba(underlying_layer) / 255.0

    # Calculate the luminosity of the background image
    background_luminosity = (rgb_to_luminosity(background_array) * 255.0).astype(UInt8)

    # Expand dimensions for further use
    foreground_rgb, background_rgb = foreground_array[:, :, :3], background_array[:, :, :3]
    foreground_a = np.expand_dims(foreground_array[:, :, 3], 2)
    background_a = np.expand_dims(background_array[:, :, 3], 2)

    # Create the blend if condition based on the luminosity range
    condition = (background_luminosity >= bg_shadows[0]) & (background_luminosity <= bg_highlights[1])
    blend_if = np.expand_dims(condition, 2)

    # Calculate the blending factors for the shadow and highlight ranges
    shadow_factor = np.interp(background_luminosity, [bg_shadows[0], bg_shadows[1]], [0, 1])
    highlight_factor = np.interp(background_luminosity, [bg_highlights[0], bg_highlights[1]], [0, 1])

    # Expand dimensions of alpha for further use
    shadow_factor = np.expand_dims(shadow_factor, 2)
    highlight_factor = np.expand_dims(highlight_factor, 2)

    blended_rgb = np.where(
        blend_if,
        foreground_rgb + (background_rgb - foreground_rgb) * (1 - shadow_factor),
        background_rgb + (foreground_rgb - background_rgb) * highlight_factor
    )
    blended_a = np.where(blend_if, foreground_a + background_a * (1 - foreground_a), background_a)

    blended_rgba = np.dstack((blended_rgb, blended_a))
    if True:
        # Scale the RGBA values back to the range [0, 255]
        blended_rgba = (blended_rgba * 255).astype(np.uint8)
        return blended_rgba

    # noinspection PyUnreachableCode
    if False:
        # Hack how to obtain T-Shirt.
        text = np.dstack((blended_rgb, fga_broadast))
        mask = foreground_a > 0.5
        background_rgb[mask] = text[:, :, :3][mask]

        background_rgba = np.dstack((background_rgb, background_a))
        return (background_rgba * 255).astype(np.uint8)


