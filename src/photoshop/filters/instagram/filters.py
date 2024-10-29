from typing import Callable

import numpy as np
from PIL import Image

from photoshop._interface import Command
from photoshop.core.dtype import Float, UInt8


class FilterMixin(object):
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

    @staticmethod
    def sepia(image: Image.Image, adj: float) -> Image.Image:
        # Store the original image mode
        image_mode = image.mode

        # Convert the image to RGBA format
        image = image.convert('RGBA')

        # Convert the image to a numpy array of float32
        # noinspection PyTypeChecker
        arr = np.array(image, dtype=np.float32)

        # Extract R, G, B channels
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        # Apply the filter formula (similar to cls.sepia or vintage filters)
        new_r = r * (1 - 0.607 * adj) + g * 0.769 * adj + b * 0.189 * adj
        new_g = r * 0.349 * adj + g * (1 - 0.314 * adj) + b * 0.168 * adj
        new_b = r * 0.272 * adj + g * 0.534 * adj + b * (1 - 0.869 * adj)

        # Assign the new color values back to the array
        arr[..., 0] = new_r
        arr[..., 1] = new_g
        arr[..., 2] = new_b

        # Clamp the values to [0, 255] and convert back to uint8
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert back to PIL image while preserving the original mode
        return Image.fromarray(arr, mode='RGBA').convert(image_mode)

    @staticmethod
    def grayscale(image: Image.Image) -> Image.Image:
        # Store the original image mode
        image_mode = image.mode

        # Convert the image to RGBA format
        image = image.convert('RGBA')

        # Convert the image to a numpy array of float32
        # noinspection PyTypeChecker
        arr = np.array(image, dtype=np.float32)

        # Calculate the cls.grayscale value and assign it to R, G, B channels
        avg = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        arr[..., 0] = arr[..., 1] = arr[..., 2] = avg

        # Clamp values to the range [0, 255] and convert back to uint8
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert back to PIL image while preserving the original mode
        return Image.fromarray(arr, mode='RGBA').convert(image_mode)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def adjust_saturation(image: Image.Image, adj: float) -> Image.Image:
        image_mode = image.mode
        image = image.convert('RGBA')

        adj = -1 if adj < -1 else adj

        # Convert the image to a numpy array
        # noinspection PyTypeChecker
        arr = np.array(image, dtype=np.float32)

        # Extract the R, G, B channels
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        # Compute the cls.grayscale value using the same weights as in the JS code
        gray = 0.2989 * r + 0.587 * g + 0.114 * b

        # Apply the saturation adjustment for each channel
        arr[..., 0] = -gray * adj + r * (1 + adj)
        arr[..., 1] = -gray * adj + g * (1 + adj)
        arr[..., 2] = -gray * adj + b * (1 + adj)

        # Clamp values to the range [0, 255] and convert back to uint8
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert back to a PIL image
        return Image.fromarray(arr).convert(image_mode)

    @staticmethod
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


class _Filters(FilterMixin):
    @classmethod
    def clarendon(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_brightness(image, .1)
        adjusted_image = cls.adjust_contrast(adjusted_image, .1)
        adjusted_image = cls.adjust_saturation(adjusted_image, .15)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def gingham(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.sepia(image, .04)
        adjusted_image = cls.adjust_contrast(adjusted_image, -.15)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def moon(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.grayscale(image)
        adjusted_image = cls.adjust_contrast(adjusted_image, -.04)
        adjusted_image = cls.adjust_brightness(adjusted_image, .1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def lark(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_brightness(image, .08)
        adjusted_image = cls.rgb_adjust(adjusted_image, [1, 1.03, 1.05])
        adjusted_image = cls.adjust_saturation(adjusted_image, .12)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def reyes(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.sepia(image, .4)
        adjusted_image = cls.adjust_brightness(adjusted_image, .13)
        adjusted_image = cls.adjust_contrast(adjusted_image, -.05)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def juno(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.rgb_adjust(image, [1.01, 1.04, 1])
        adjusted_image = cls.adjust_saturation(adjusted_image, .3)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def slumber(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_brightness(image, .1)
        adjusted_image = cls.adjust_saturation(adjusted_image, -.5)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def crema(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply RGB adjustment
        adjusted_image = cls.rgb_adjust(image, [1.04, 1, 1.02])
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(adjusted_image, -0.05)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def ludwig(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply brightness adjustment
        adjusted_image = cls.adjust_brightness(image, 0.05)
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(adjusted_image, -0.03)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def aden(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply color filter
        adjusted_image = cls.color_filter(image, [228, 130, 225], 0.13)
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(adjusted_image, -0.2)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def perpetua(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply RGB adjustment
        adjusted_image = cls.rgb_adjust(image, [1.05, 1.1, 1.0])
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def amaro(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(image, 0.3)
        # Apply brightness adjustment
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.15)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def mayfair(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply color filter
        adjusted_image = cls.color_filter(image, [230, 115, 108], 0.05)
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(adjusted_image, 0.15)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def rise(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply color filter
        adjusted_image = cls.color_filter(image, [255, 170, 0], 0.1)
        # Apply brightness adjustment
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.09)
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(adjusted_image, 0.1)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def valencia(cls, image: Image.Image, strength: float = 1.0):
        adjusted_image = cls.color_filter(image, [255, 225, 80], .08)
        adjusted_image = cls.adjust_saturation(adjusted_image, .1)
        adjusted_image = cls.adjust_contrast(adjusted_image, .05)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def xpro2(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 255, 0], .07)
        adjusted_image = cls.adjust_saturation(adjusted_image, .2)
        adjusted_image = cls.adjust_contrast(adjusted_image, .15)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def sierra(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_contrast(image, -.15)
        adjusted_image = cls.adjust_saturation(adjusted_image, .1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def willow(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply cls.grayscale filter
        adjusted_image = cls.grayscale(image)
        # Apply color filter
        adjusted_image = cls.color_filter(adjusted_image, [100, 28, 210], 0.03)
        # Apply brightness adjustment
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.1)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def lofi(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_contrast(image, 0.15)
        adjusted_image = cls.adjust_saturation(adjusted_image, 0.2)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def inkwell(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply cls.grayscale filter
        adjusted_image = cls.grayscale(image)
        # Apply strength blending (no blending needed here since strength is not adjustable)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def nashville(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply color filter
        adjusted_image = cls.color_filter(image, [220, 115, 188], 0.12)
        # Apply contrast adjustment
        adjusted_image = cls.adjust_contrast(adjusted_image, -0.05)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    # Other than intagram
    @classmethod
    def skyline(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_saturation(image, .35)
        adjusted_image = cls.adjust_brightness(adjusted_image, .1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def hefe(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        # Apply contrast adjustment
        adjusted_image = cls.adjust_contrast(image, 0.1)
        # Apply saturation adjustment
        adjusted_image = cls.adjust_saturation(adjusted_image, 0.15)
        # Apply strength blending
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def hudson(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.rgb_adjust(image, [1, 1, 1.25])
        adjusted_image = cls.adjust_contrast(adjusted_image, 0.1)
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.15)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def stinson(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_brightness(image, 0.1)
        adjusted_image = cls.sepia(adjusted_image, 0.3)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def vesper(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 225, 0], 0.05)
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.06)
        adjusted_image = cls.adjust_contrast(adjusted_image, 0.06)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def earlybird(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 165, 40], 0.2)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def brannan(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_contrast(image, 0.2)
        adjusted_image = cls.color_filter(adjusted_image, [140, 10, 185], 0.1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def sutro(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_brightness(image, -0.1)
        adjusted_image = cls.adjust_saturation(adjusted_image, -0.1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def toaster(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.sepia(image, 0.1)
        adjusted_image = cls.color_filter(adjusted_image, [255, 145, 0], 0.2)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def walden(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_brightness(image, 0.1)
        adjusted_image = cls.color_filter(adjusted_image, [255, 255, 0], 0.2)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def f1977(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 25, 0], 0.15)
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def kelvin(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 140, 0], 0.1)
        adjusted_image = cls.rgb_adjust(adjusted_image, [1.15, 1.05, 1])
        adjusted_image = cls.adjust_saturation(adjusted_image, 0.35)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def maven(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [225, 240, 0], 0.1)
        adjusted_image = cls.adjust_saturation(adjusted_image, 0.25)
        adjusted_image = cls.adjust_contrast(adjusted_image, 0.05)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def ginza(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.sepia(image, 0.06)
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def dogpatch(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.adjust_contrast(image, 0.15)
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def brooklyn(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [25, 240, 252], 0.05)
        adjusted_image = cls.sepia(adjusted_image, 0.3)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def helena(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [208, 208, 86], 0.2)
        adjusted_image = cls.adjust_contrast(adjusted_image, 0.15)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def ashby(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 160, 25], 0.1)
        adjusted_image = cls.adjust_brightness(adjusted_image, 0.1)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    @classmethod
    def charmes(cls, image: Image.Image, strength: float = 1.0) -> Image.Image:
        adjusted_image = cls.color_filter(image, [255, 50, 80], 0.12)
        adjusted_image = cls.adjust_contrast(adjusted_image, 0.05)
        return cls._apply_strength(image, adjusted_image, strength=strength)

    
class FilterReceiver(object):
    @staticmethod
    def transform(rgba: np.ndarray, strength: Float, filter_fn: Callable):
        rgba_ = Image.fromarray(rgba.astype(UInt8)).convert('RGBA')
        transformed_: Image.Image = filter_fn(rgba_, strength)
        # noinspection PyTypeChecker
        return np.array(transformed_).astype(UInt8)


class Filter(Command):
    def __init__(
            self,
            receiver: FilterReceiver,
            rgba: np.ndarray,
            filter_: str,
            strength: Float = 1.0,
    ):
        super(Filter, self).__init__()

        self._rgba = rgba.copy()
        self._receiver = receiver

        self._strength = strength
        self._filter = filter_
        self._filter_fn = getattr(_Filters, filter_)

    def execute(self) -> None:
        self._result = self._receiver.transform(self._rgba, self._strength, self._filter_fn)

    def validator(self):
        if self._filter is None:
            raise ValueError(f"Invalid filter with name `{self._filter}`")

        if not isinstance(self._rgba, np.ndarray) or self._rgba.dtype not in (UInt8,):
            raise TypeError('Invalid image type. Expected numpy.uint8 in shape h x w x 4')


def instagram__(rgba: np.ndarray, strength: Float = 1.0, filter_: str = None):
    receiver = FilterReceiver()
    command = Filter(receiver, rgba, filter_, strength)
    command.execute()
    return command.result
