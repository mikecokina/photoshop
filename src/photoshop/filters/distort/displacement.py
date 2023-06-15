from PIL import Image

from ..._interface import Command
from ...libs.numpy import np
from ...core.dtype import UInt8, Float
from ...ops.transform import expand_as_rgba


class DisplacementReceiver(object):
    @classmethod
    def _displacement(cls, rgba: np.ndarray, displacement_map: np.ndarray, strength: Float) -> np.ndarray:
        """
        Compute displacement of image based on supplied displacement map.
        Mimic Photoshop: Filter -> Distort -> Displacement

        :param rgba: np.ndarray; Image of shape (h x w x 3) or (h x w x 4)
        :param displacement_map:  Image of shape (h x w x 3)
        :param strength: Float; strength of displacemepnt
        :return: np.ndaray; Image of shape (h x w x 4)
        """
        # TODO: sanitize input (image and map should have same w x h; or resize properly)
        # Extend image with alpha channel if not presented
        rgba = expand_as_rgba(rgba).astype(UInt8)

        # Extract width and height from image
        height, width = rgba.shape[:2]

        # Convert the images to grayscale
        # noinspection PyTypeChecker
        displacement_map_gray = np.array(Image.fromarray(displacement_map).convert('L'), dtype=UInt8)

        # Get the pixel data as numpy arrays
        displacement_map_array = np.array(displacement_map_gray, dtype=UInt8)

        # Normalize the displacement map values to the range [-1, 1]
        displacement_map_normalized = (displacement_map_array / 255.0) * 2 - 1

        # Create a new empty image to store the displaced pixels
        result_image = Image.new('RGBA', (width, height))
        # noinspection PyTypeChecker
        result_array = np.array(result_image)

        # Iterate over each pixel in the image
        for y in range(height):
            for x in range(width):
                # Calculate the displacement amount based on the grayscale value
                displacement = displacement_map_normalized[y, x] * strength

                # Calculate the new pixel coordinates with displacement
                new_x = x + displacement
                new_y = y + displacement

                # Interpolate the pixel value from the original image
                interpolated_pixel = np.zeros(4)
                if 0 <= new_x < width - 1 and 0 <= new_y < height - 1:
                    # Use bilinear interpolation to calculate the new pixel value
                    x_floor, y_floor = int(new_x), int(new_y)
                    x_ceil, y_ceil = x_floor + 1, y_floor + 1
                    x_weight, y_weight = new_x - x_floor, new_y - y_floor

                    # Interpolate each color channel
                    for ch in range(4):
                        top = rgba[y_floor, x_floor, ch] * (1 - x_weight) + rgba[y_floor, x_ceil, ch] * x_weight
                        bottom = rgba[y_ceil, x_floor, ch] * (1 - x_weight) + rgba[y_ceil, x_ceil, ch] * x_weight
                        interpolated_pixel[ch] = top * (1 - y_weight) + bottom * y_weight

                # Assign the interpolated pixel value to the result image
                result_array[y, x] = interpolated_pixel.astype(UInt8)

        return result_array

    def transform(self, rgba: np.ndarray, displacement_map: np.ndarray, strength: Float) -> np.ndarray:
        return self._displacement(rgba, displacement_map, strength)


class Displacement(Command):
    def __init__(
            self,
            receiver: DisplacementReceiver,
            rgba: np.ndarray,
            displacement_map: np.ndarray,
            strength: Float
    ) -> None:
        super(Displacement, self).__init__()

        self._rgba = rgba.copy()
        self._receiver = receiver
        self._displacement_map = displacement_map
        self._strength = strength
        self._result = None

    def execute(self) -> None:
        self._result = self._receiver.transform(self._rgba, self._displacement_map, self._strength)


def displacement__(rgba: np.ndarray, displacement_map: np.ndarray, strength: Float) -> np.ndarray:
    """
    Compute displacement of image based on supplied displacement map.
    Mimic Photoshop: Filter -> Distort -> Displacement

    :param rgba: np.ndarray; Image of shape (h x w x 3) or (h x w x 4)
    :param displacement_map:  Image of shape (h x w x 3)
    :param strength: Float; strength of displacemepnt
    :return: np.ndaray; Image of shape (h x w x 4)
    """
    receiver = DisplacementReceiver()
    command = Displacement(rgba=rgba, receiver=receiver, displacement_map=displacement_map, strength=strength)
    command.execute()
    return command.result
