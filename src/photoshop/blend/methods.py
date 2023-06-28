from ..core.dtype import Float, UInt8
from ..libs.numpy import np
from ..ops.transform import expand_as_rgba


def _compose_alpha(
        foreground: np.ndarray,
        background: np.ndarray,
        opacity: Float
) -> np.ndarray:
    """
    Kudos to GitHub: `flrs/blend_modes`.

    :param foreground: np.ndarray;
    :param background: np.ndarray;
    :param opacity: Float;
    :return: np.ndarray;
    """
    comp_alpha = np.minimum(foreground[:, :, 3], background[:, :, 3]) * opacity
    new_alpha = foreground[:, :, 3] + (1.0 - foreground[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.nan] = 0.0
    return ratio


def normal(
        foreground: np.ndarray,
        background: np.ndarray,
        opacity: Float = 1.0,
) -> np.ndarray:
    """
    Provides normal blending.

    :param opacity: Float;
    :param foreground: np.ndarray; numpy array of shape (h, w, 4)
    :param background: np.ndarray; numpy array of shape (h, w, 4)
    :return: np.ndarray; numpy array of shape (h, w, 3)
    """
    foreground = expand_as_rgba(foreground) / 255.0
    background = expand_as_rgba(background) / 255.0

    foreground_a = np.expand_dims(foreground[:, :, 3], 2)
    background_a = np.expand_dims(background[:, :, 3], 2)

    # Calculate the resulting alpha after blending
    result_alpha = foreground_a + background_a * (1 - foreground_a)

    # Calculate the blended RGB values
    np.seterr(divide='ignore', invalid='ignore')
    result_rgb = (
            (foreground[:, :, :3] * foreground_a * opacity
             + background[:, :, :3] * background_a * (1 - foreground_a))
            / result_alpha
    )
    result_rgb[result_rgb == np.nan] = 0.0
    # Combine the RGB values with the resulting alpha
    result = np.dstack((result_rgb, result_alpha[..., 0]))

    return (result * 255).astype(UInt8)


def screen(
        foreground: np.ndarray,
        background: np.ndarray,
        opacity: Float = 1.0,
) -> np.ndarray:
    foreground_array = expand_as_rgba(foreground) / 255.0
    background_array = expand_as_rgba(background) / 255.0

    ratio = _compose_alpha(foreground_array, background_array, opacity)
    comp = 1.0 - (1.0 - foreground_array[:, :, :3]) * (1.0 - background_array[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + foreground_array[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, foreground_array[:, :, 3])))  # add alpha channel and replace nans
    return (img_out * 255).astype(UInt8)


def multiply(
        foreground: np.ndarray,
        background: np.ndarray,
        opacity: Float = 1.0,
) -> np.ndarray:
    foreground = expand_as_rgba(foreground) / 255.0
    background = expand_as_rgba(background) / 255.0

    ratio = _compose_alpha(foreground, background, opacity)

    comp = np.clip(background[:, :, :3] * foreground[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + foreground[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, foreground[:, :, 3])))  # add alpha channel and replace nans
    return (img_out * 255).astype(UInt8)
