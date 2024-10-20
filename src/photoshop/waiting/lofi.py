from PIL import Image
import numpy as np


class AbstractFilter(object):
    @staticmethod
    def _apply_strength(image: Image.Image, adjusted_image: Image.Image, strength: float) -> Image.Image:
        # Ensure strength is clamped between 0 and 1
        strength = max(0., min(1., strength))

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


class Filters(AbstractFilter):
    @classmethod
    def lofi(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = adjust_contrast(image, 0.15)
        adjusted_image = adjust_saturation(adjusted_image, 0.2)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def valencia(cls, image: Image.Image, strength: float = 1.0):
        adjusted_image = color_filter(image, [255, 225, 80], .08)
        adjusted_image = adjust_saturation(adjusted_image, .1)
        adjusted_image = adjust_contrast(adjusted_image, .05)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def xpro2(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = color_filter(image, [255, 255, 0], .07)
        adjusted_image = adjust_saturation(adjusted_image, .2)
        adjusted_image = adjust_contrast(adjusted_image, .15)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def sierra(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = adjust_contrast(image, -.15)
        adjusted_image = adjust_saturation(adjusted_image, .1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def skyline(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = adjust_saturation(image, .35)
        adjusted_image = adjust_brightness(adjusted_image, .1)
        return cls._apply_strength(image, adjusted_image, strength=strength)


    @classmethod
    def lark(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = adjust_brightness(image, .08)
        adjusted_image = rgb_adjust(adjusted_image, [1, 1.03, 1.05])
        adjusted_image = adjust_saturation(adjusted_image, .12)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def moon(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = grayscale(image)
        adjusted_image = adjust_contrast(adjusted_image, -.04)
        adjusted_image = adjust_brightness(adjusted_image, .1)
        return cls._apply_strength(image, adjusted_image, strength=strength)


def grayscale(image: Image.Image) -> Image.Image:
    # Store the original image mode
    image_mode = image.mode

    # Convert the image to RGBA format
    image = image.convert('RGBA')

    # Convert the image to a numpy array of float32
    # noinspection PyTypeChecker
    arr = np.array(image, dtype=np.float32)

    # Calculate the grayscale value and assign it to R, G, B channels
    avg = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    arr[..., 0] = arr[..., 1] = arr[..., 2] = avg

    # Clamp values to the range [0, 255] and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert back to PIL image while preserving the original mode
    return Image.fromarray(arr, mode='RGBA').convert(image_mode)


def color_filter(image: Image.Image, rgb_color: list, adj: float) -> Image.Image:
    image_mode = image.mode
    image = image.convert('RGBA')

    # Convert the image to a numpy array
    # noinspection PyTypeChecker
    arr = np.array(image, dtype=np.float32)

    # Apply the color filter adjustment to R, G, B channels
    arr[..., 0] -= (arr[..., 0] - rgb_color[0]) * adj  # Red channel
    arr[..., 1] -= (arr[..., 1] - rgb_color[1]) * adj  # Green channel
    arr[..., 2] -= (arr[..., 2] - rgb_color[2]) * adj  # Blue channel

    # Clamp the values to the valid range [0, 255] and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(arr).convert(image_mode)


def adjust_brightness(image: Image.Image, adj: float):
    image_mode = image.mode
    image = image.convert('RGBA')

    # Clamp adj to range [-1, 1] and convert it to a 0-255 range adjustment
    adj = max(-1., min(1., adj))
    adj = int(255 * adj)

    # Convert the image to a numpy array
    # noinspection PyTypeChecker
    arr = np.array(image, dtype=np.float32)

    # Adjust the R, G, B channels by adding the adj value
    arr[..., 0:3] += adj

    # Clamp the values to the valid range [0, 255] and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(arr).convert(image_mode)


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


def rgb_adjust(image: Image.Image, rgb_adj: list) -> Image.Image:
    # Store the original image mode
    image_mode = image.mode

    # Convert the image to RGBA format
    image = image.convert('RGBA')

    # Convert the image to a numpy array of float32
    # noinspection PyTypeChecker
    arr = np.array(image, dtype=np.float32)

    # Apply the RGB adjustments
    arr[..., 0] *= rgb_adj[0]  # Red channel
    arr[..., 1] *= rgb_adj[1]  # Green channel
    arr[..., 2] *= rgb_adj[2]  # Blue channel

    # Clamp values to the range [0, 255] and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert back to PIL image while preserving the original mode
    return Image.fromarray(arr, mode='RGBA').convert(image_mode)


# Usage example
image_ = Image.open("/home/mike/Downloads/PXL_20241019_160706455_EDIT.png")
lofi_image = Filters.moon(image_, strength=1.0)
lofi_image.show()  # Display the modified image
