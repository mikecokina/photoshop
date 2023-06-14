from typing import Tuple

from ..core.dtype import Int, UInt8, Float32
from ..libs.numpy import np
from ..ops.transform import rgb_to_luminosity, expand_as_rgba


def _normal_blend(fg_rgb: np.ndarray, fg_a: np.ndarray, bg_rgb: np.ndarray, bg_a: np.ndarray) -> np.ndarray:
    """
    Provides normal blending.
    Do not use outside of this scope.

    :param fg_rgb: np.ndarray; numpy array of shape (h, w, 3)
    :param fg_a: np.ndarray; numpy array of shape (h, w) or (h, w, 1)
    :param bg_rgb: np.ndarray; numpy array of shape (h, w, 3)
    :param bg_a: np.ndarray; numpy array of shape (h, w) or (h, w, 1)
    :return: np.ndarray; numpy array of shape (h, w, 3)
    """
    fg_a = np.expand_dims(fg_a, 2) if len(fg_a.shape) == 2 else fg_a
    bg_a = np.expand_dims(bg_a, 2) if len(bg_a.shape) == 2 else bg_a
    return fg_rgb * fg_a + bg_rgb * bg_a * (1 - fg_a)


def _normal_blend_if(
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


def _sanitize_inputs(
    underlying_layer_shadows_range: Tuple[Int, Int],
    underlying_layer_highlights_range: Tuple[Int, Int]
) -> Tuple[Tuple[Int, Int], Tuple[Int, Int]]:
    """
    Sanitize input for bonditional complex blending.

    :param underlying_layer_shadows_range: Tuple[Int, Int];
    :param underlying_layer_highlights_range: Tuple[Int, Int];
    :return: Tuple[Tuple[Int, Int], Tuple[Int, Int]];
    """
    if underlying_layer_shadows_range[0] > underlying_layer_shadows_range[1]:
        print("Invalid shadows")
        underlying_layer_shadows_range = (underlying_layer_shadows_range[0], underlying_layer_shadows_range[0])
    if underlying_layer_highlights_range[0] > underlying_layer_highlights_range[1]:
        print("Invalid highlights")
        underlying_layer_highlights_range = (underlying_layer_highlights_range[1], underlying_layer_highlights_range[1])
    if underlying_layer_shadows_range[1] >= underlying_layer_highlights_range[0]:
        msg = "Unsupported options: `underlying_layer_shadows_range[1]` cannot " \
              "be higher than `underlying_layer_highlights_range[0]`."
        raise ValueError(msg)

    return underlying_layer_shadows_range, underlying_layer_highlights_range


def normal_blend_if(
        this_layer: np.ndarray,
        underlying_layer: np.ndarray,
        underlying_layer_shadows_range: Tuple[Int, Int] = (0, 0),
        underlying_layer_highlights_range: Tuple[Int, Int] = (255, 255)
) -> np.ndarray:
    """
    Simple implementation for coditional normal blending as implemented in photoshop.
    Does not allow overlaping of ranges as Photoshop does, myabe will be solved in the future.

    :param this_layer: np.ndarray; Image of shape (h, w, 3) or (h, w, 4)
    :param underlying_layer: np.ndarray; Image of shape (h, w, 3) or (h, w, 4)
    :param underlying_layer_shadows_range: Tuple[Int, Int];
    :param underlying_layer_highlights_range: Tuple[Int, Int]
    :return: np.ndarray; Image of shape (h, w, 3)
    """
    # To avoid any mutation on original inputs
    this_layer, underlying_layer = this_layer.copy(), underlying_layer.copy()

    # Input validation.
    bg_shadows, bg_highlights = _sanitize_inputs(underlying_layer_shadows_range, underlying_layer_highlights_range)

    # Expand with alpha if missing
    this_layer = expand_as_rgba(this_layer) / 255.0
    underlying_layer = expand_as_rgba(underlying_layer) / 255.0

    # Calculate the luminosity of the background image
    bg_luminosity = (rgb_to_luminosity(underlying_layer) * 255.0).astype(UInt8)

    # Expand dimensions for further use
    fg_rgb, bg_rgb = this_layer[:, :, :3], underlying_layer[:, :, :3]
    fg_a, bg_a = this_layer[:, :, 3], underlying_layer[:, :, 3]

    # Define result image
    blended_rgb = np.ones(bg_rgb.shape).astype(Float32)

    # Determine blending conditions
    # ### Out of range
    bg_condition = (bg_luminosity < bg_shadows[0]) | (bg_luminosity > bg_highlights[1])
    # ### Middle range
    blending_condition = (bg_luminosity >= bg_shadows[1]) & (bg_luminosity <= bg_highlights[0])
    # ### Shadows range
    shadows_condition = (bg_luminosity >= bg_shadows[0]) & (bg_luminosity <= bg_shadows[1])
    # ### Highlights range
    highlights_condition = (bg_luminosity >= bg_highlights[0]) & (bg_luminosity <= bg_highlights[1])

    # Calculate the blending factors for the shadow and highlight ranges
    shadow_factor = np.interp(bg_luminosity, [bg_shadows[0], bg_shadows[1]], [0, 1])
    highlight_factor = np.interp(bg_luminosity, [bg_highlights[0], bg_highlights[1]], [0, 1])

    # Blending (replace pixels with bakground pixels) out of codnitions range
    blended_rgb[bg_condition] = bg_rgb[bg_condition]

    # Blending within middle range
    blended_ = _normal_blend(fg_rgb, fg_a, bg_rgb, bg_a)
    blended_rgb[blending_condition] = blended_[blending_condition]

    # Blending within shadows range
    fg_a[shadows_condition] = (fg_a * shadow_factor)[shadows_condition]
    shadows_blended_ = _normal_blend(fg_rgb, fg_a, bg_rgb, bg_a)
    blended_rgb[shadows_condition] = shadows_blended_[shadows_condition]

    # Blending within highlights range
    fg_a[highlights_condition] = (fg_a * (1 - highlight_factor))[highlights_condition]
    highlights_blended_ = _normal_blend(fg_rgb, fg_a, bg_rgb, bg_a)
    blended_rgb[highlights_condition] = highlights_blended_[highlights_condition]

    # Transform image to Uint8 form
    return (blended_rgb * 255).astype(UInt8)
