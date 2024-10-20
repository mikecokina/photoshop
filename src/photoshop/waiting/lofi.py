from PIL import Image
import numpy as np


def lofi(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Apply the Lo-Fi filter with a given strength.
    :param image: The input PIL Image.
    :param strength: Strength of the filter from 0.0 (no filter) to 1.0 (full filter).
    :return: The filtered PIL Image.
    """
    # Ensure strength is clamped between 0 and 1
    strength = max(0., min(1., strength))

    # Apply contrast and saturation adjustments
    adjusted_image = adjust_contrast(image, 0.15)
    adjusted_image = adjust_saturation(adjusted_image, 0.2)

    if strength < 1.0:
        # Blend between the original and adjusted image based on strength
        # noinspection PyTypeChecker
        adjusted_arr = np.array(adjusted_image.convert('RGBA'), dtype=np.float32)
        # noinspection PyTypeChecker
        original_arr = np.array(image.convert('RGBA'), dtype=np.float32)

        # Interpolate between original and adjusted based on strength
        blended_arr = (1 - strength) * original_arr + strength * adjusted_arr
        blended_arr = np.clip(blended_arr, 0, 255).astype(np.uint8)

        # Convert back to PIL image
        return Image.fromarray(blended_arr).convert(image.mode)

    return adjusted_image


def adjust_contrast(image: Image.Image, adj: float) -> Image.Image:
    image_mode = image.mode
    image = image.convert('RGBA')

    adj *= 255
    factor = 259 * (adj + 255) / (255 * (259 - adj))
    # noinspection PyTypeChecker
    arr = np.array(image, dtype=np.float32)

    arr[..., 0:3] = factor * (arr[..., 0:3] - 128) + 128

    # Clamp values to the range [0, 255] and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    return Image.fromarray(arr).convert(image_mode)


def adjust_saturation(image: Image.Image, adj: float) -> Image.Image:
    image_mode = image.mode
    image = image.convert('RGBA')

    adj = -1 if adj < -1 else adj

    # Convert the image to a numpy array
    # noinspection PyTypeChecker
    arr = np.array(image, dtype=np.float32)

    # Extract the R, G, B channels
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Compute the grayscale value using the same weights as in the JS code
    gray = 0.2989 * r + 0.587 * g + 0.114 * b

    # Apply the saturation adjustment for each channel
    arr[..., 0] = -gray * adj + r * (1 + adj)
    arr[..., 1] = -gray * adj + g * (1 + adj)
    arr[..., 2] = -gray * adj + b * (1 + adj)

    # Clamp values to the range [0, 255] and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert back to a PIL image
    return Image.fromarray(arr).convert(image_mode)


# Usage example
image_ = Image.open("/home/mike/Downloads/PXL_20241019_160706455_EDIT.png")
lofi_image = lofi(image_, strength=1.0)
lofi_image.show()  # Display the modified image
