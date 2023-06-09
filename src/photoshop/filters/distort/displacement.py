import numpy as np
from PIL import Image


def displacement_filter(image, displacement_map, strength):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Convert the images to grayscale
    displacement_map_gray = displacement_map.convert('L')

    # Get the pixel data as numpy arrays
    image_array = np.array(image)
    displacement_map_array = np.array(displacement_map_gray)

    # Normalize the displacement map values to the range [-1, 1]
    displacement_map_normalized = (displacement_map_array / 255.0) * 2 - 1

    # Create a new empty image to store the displaced pixels
    result_image = Image.new('RGBA', image.size)
    # noinspection PyTypeChecker
    result_array = np.array(result_image)

    # Iterate over each pixel in the image
    for y in range(image.height):
        for x in range(image.width):
            # Calculate the displacement amount based on the grayscale value
            displacement = displacement_map_normalized[y, x] * strength

            # Calculate the new pixel coordinates with displacement
            new_x = x + displacement
            new_y = y + displacement

            # Interpolate the pixel value from the original image
            interpolated_pixel = np.zeros(4)
            if 0 <= new_x < image.width - 1 and 0 <= new_y < image.height - 1:
                # Use bilinear interpolation to calculate the new pixel value
                x_floor = int(new_x)
                y_floor = int(new_y)
                x_ceil = x_floor + 1
                y_ceil = y_floor + 1
                x_weight = new_x - x_floor
                y_weight = new_y - y_floor

                # Interpolate each color channel
                for channel in range(4):
                    top = image_array[y_floor, x_floor, channel] * (1 - x_weight) + image_array[y_floor, x_ceil, channel] * x_weight
                    bottom = image_array[y_ceil, x_floor, channel] * (1 - x_weight) + image_array[y_ceil, x_ceil, channel] * x_weight
                    interpolated_pixel[channel] = top * (1 - y_weight) + bottom * y_weight

            # Assign the interpolated pixel value to the result image
            result_array[y, x] = interpolated_pixel.astype(int)

    result_image = Image.fromarray(result_array)
    return result_image


def main():
    # Example usage
    image = Image.open('../../../../text.png')
    displacement_map = Image.open('../../../../map.png')
    background = Image.open('../../../../text.png')

    filtered_image = displacement_filter(image, displacement_map, strength=10)
    filtered_image.show()

    filtered_image.save('../../../../warp.png')


if __name__ == '__main__':
    main()
