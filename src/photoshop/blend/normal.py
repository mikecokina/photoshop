"""
Hello,

I am currently working on conditional blending implementation like one used in
Photoshop -> Blending Options -> Blend If. Anyone. who is a bit familiar with Photoshop should
be aware of such functionality. For simplicity, lets assume `Blend If` implementation just for Underlaying
image. Anyway, it is most important to me. Photoshop feature works in two modes. First, simplier mode, take care
for two values, actually shadows and highlight. Blending among layer (foreground) on top with underlying layer
(background) is executed if grayscale pixels of background are between interval [shadow, highlight], otherwise
original background pixels are taken. I implemented such behavior in function bellow. What is crucial for me,
is more complex implementation, where shadow and highlight thresholds are split into two values. It eventualy leads
to smooth blending, but I have no idea what is going on even though, I watched several tutorial about Photoshop.
It seems like additional blending is executed based on scale defined by shadow/highlight range, but have no idea how
and how to add it to my algorithm. The closest implementation I was able to derive is second snippet,
but it doesn't work properly if foreground image has its own mask already on the input.
Here are also images I blend as expected results and my results.

Both functions take input images in form of 0 - 255 uint8 numpy array in shape (h, w, ch).

Can anyone help with this issue?
Thank You
"""


from typing import Tuple
import numpy as np


def expand_as_rgba(image: np.ndarray) -> np.ndarray:
    # Add alpha-channels, if they are not provided
    if image.shape[2] == 3:
        return np.dstack((image, np.ones(image.shape[:2] + (1,)) * 255)).astype(np.uint8)
    return image


def normal_blend_if(
        this_layer: np.ndarray,
        underlying_layer: np.ndarray,
        underlying_layer_shadows_range: Tuple = (0, 0),
        underlying_layer_highlights_range: Tuple = (255, 255),
) -> np.ndarray:
    bg_shadows, bg_highlights = underlying_layer_shadows_range, underlying_layer_highlights_range

    # Expand with alpha if missing
    foreground_array = expand_as_rgba(this_layer) / 255.0
    background_array = expand_as_rgba(underlying_layer) / 255.0

    # Extract the individual channels (R, G, B, A)
    foreground_r, foreground_g, foreground_b, foreground_a = np.rollaxis(foreground_array, axis=-1)
    background_r, background_g, background_b, background_a = np.rollaxis(background_array, axis=-1)

    # Calculate the luminosity of the background image
    background_luminosity = (0.299 * background_r + 0.587 * background_g + 0.114 * background_b) * 255.0

    # Create the blend if condition based on the luminosity range
    blend_if = (background_luminosity >= bg_shadows[0]) & (background_luminosity <= bg_highlights[1])
    blend_if_broadcast = np.expand_dims(blend_if, 2)

    foreground_rgb, background_rgb = foreground_array[:, :, :3], background_array[:, :, :3]
    fga_broadast, bga_broadcast = np.expand_dims(foreground_a, 2), np.expand_dims(background_a, 2)

    # Conditional blending
    blended_rgb = np.where(
        blend_if_broadcast,
        (foreground_rgb * fga_broadast + background_rgb * bga_broadcast * (1 - fga_broadast)) / (
                fga_broadast + bga_broadcast * (1 - fga_broadast)),
        background_rgb
    )

    blended_a = np.where(
        blend_if_broadcast,
        fga_broadast + bga_broadcast * (1 - fga_broadast),
        bga_broadcast
    )

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
    bg_shadows, bg_highlights = underlying_layer_shadows_range, underlying_layer_highlights_range

    # Expand with alpha if missing
    foreground_array = expand_as_rgba(this_layer) / 255.0
    background_array = expand_as_rgba(underlying_layer) / 255.0

    # Extract the individual channels (R, G, B, A)
    foreground_r, foreground_g, foreground_b, foreground_a = np.rollaxis(foreground_array, axis=-1)
    background_r, background_g, background_b, background_a = np.rollaxis(background_array, axis=-1)

    # Calculate the luminosity of the background image
    background_luminosity = (0.299 * background_r + 0.587 * background_g + 0.114 * background_b) * 255.0

    # Create the blend if condition based on the luminosity range
    blend_if = (background_luminosity >= bg_shadows[0]) & (background_luminosity <= bg_highlights[1])
    blend_if_broadcast = np.expand_dims(blend_if, 2)

    # Calculate the blending factors for the shadow and highlight ranges
    shadow_factor = np.interp(background_luminosity, [bg_shadows[0], bg_shadows[1]], [0, 1])
    highlight_factor = np.interp(background_luminosity, [bg_highlights[0], bg_highlights[1]], [0, 1])

    # Expand dimensions of alpha for further use
    fga_broadast, bga_broadcast = np.expand_dims(foreground_a, 2), np.expand_dims(background_a, 2)
    foreground_rgb, background_rgb = foreground_array[:, :, :3], background_array[:, :, :3]

    shadow_factor = np.expand_dims(shadow_factor, 2)
    highlight_factor = np.expand_dims(highlight_factor, 2)

    blended_rgb = np.where(
        blend_if_broadcast,
        foreground_rgb + (background_rgb - foreground_rgb) * (1 - shadow_factor),
        background_rgb + (foreground_rgb - background_rgb) * highlight_factor
    )

    blended_a = np.where(
        blend_if_broadcast,
        fga_broadast + bga_broadcast * (1 - fga_broadast),
        bga_broadcast
    )

    blended_rgba = np.dstack((blended_rgb, blended_a))
    # # Scale the RGBA values back to the range [0, 255]
    # blended_rgba = (blended_rgba * 255).astype(np.uint8)
    # return blended_rgba

    # Hack how to obtain T-Shirt.
    text = np.dstack((blended_rgb, fga_broadast))
    mask = foreground_a > 0.5
    background_rgb[mask] = text[:, :, :3][mask]

    background_rgba = np.dstack((background_rgb, background_a))
    return (background_rgba * 255).astype(np.uint8)


