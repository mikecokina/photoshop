from photoshop.blend.normal import normal_complex_blend_if
from photoshop.libs.numpy import np
from photoshop.core.dtype import UInt8
from PIL import Image

from photoshop.ops.transform import expand_as_rgba


def displacement_filter(image: np.ndarray, displacement_map: np.ndarray, strength: float) -> np.ndarray:
    # Extend image with alpha channel if not presented
    image = expand_as_rgba(image).astype(UInt8)

    # Extract width and height from image
    height, width = image.shape[:2]

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
                    top = image[y_floor, x_floor, ch] * (1 - x_weight) + image[y_floor, x_ceil, ch] * x_weight
                    bottom = image[y_ceil, x_floor, ch] * (1 - x_weight) + image[y_ceil, x_ceil, ch] * x_weight
                    interpolated_pixel[ch] = top * (1 - y_weight) + bottom * y_weight

            # Assign the interpolated pixel value to the result image
            result_array[y, x] = interpolated_pixel.astype(UInt8)

    return result_array


# noinspection PyUnreachableCode
def main():
    # Example usage
    import cv2  # import OpenCV
    import numpy

    # background_img_float = cv2.imread('../../../../base.jpg', -1).astype(float)
    # foreground_img_float = cv2.imread('../../../../text.png', -1).astype(float)
    # blended_img_float = normal(background_img_float, foreground_img_float, opacity=1)
    # blended_img_uint8 = blended_img_float.astype(numpy.uint8)
    # cv2.imshow('window', blended_img_uint8)
    # cv2.waitKey()  # Press a key to close window with the image.

    if False:
        layer = np.asarray(Image.open('../../../../text.png').convert('RGBA'), dtype=UInt8)  # noqa
        displacement_map = np.asarray(Image.open('../../../../map.png'), dtype=UInt8)  # noqa
        underlying_layer = np.asarray(Image.open('../../../../base.jpg').convert('RGBA'), dtype=UInt8) # noqa

        filtered_image = displacement_filter(layer, displacement_map, strength=10)

        Image.fromarray(filtered_image).save('../../../../text_warp.png')

        blended = normal_complex_blend_if(
            filtered_image,
            underlying_layer,
            underlying_layer_shadows_range=(0, 50),
            underlying_layer_highlights_range=(255, 255)
        )
        Image.fromarray(blended).show()

    if True:
        foreground = (np.ones((500, 500, 3))).astype(UInt8)
        foreground[:, :, 1] = foreground[:, :, 1] * 255
        # noinspection PyTypeChecker
        underlying_layer = np.array(Image.open('../../../../foreground.png').convert('RGBA'))
        blended = normal_complex_blend_if(
            foreground,
            underlying_layer,
            underlying_layer_shadows_range=(0, 0),
            underlying_layer_highlights_range=(250, 255)
        )
        Image.fromarray(blended).show()


if __name__ == '__main__':
    main()
