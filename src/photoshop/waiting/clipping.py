from pathlib import Path
from typing import Union, Tuple, List

import scipy.ndimage as ndi
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

COLOR_FILTER_THRESHOLD_MID = 128
COLOR_FILTER_THRESHOLD_MID_HIGH = 192
COLOR_FILTER_VALUE_HIGH = 255
COLOR_FILTER_VALUE_LOW = 0
COLOR_FILTER_VALUE_MID = 64
TARGET_PERCENTAGE = 20

def histogram(image: Image.Image, ignore_color: Union[List[int], int] = None):
    # Convert image to numpy array
    # noinspection PyTypeChecker
    image_data = np.array(image)

    # If a color should be ignored, filter it out
    if ignore_color is not None:
        if isinstance(ignore_color, int):
            image_data = image_data[image_data != ignore_color]
        else:
            for ignore_color_ in ignore_color:
                image_data = image_data[image_data != ignore_color_]

    # Calculate total number of pixels after filtering
    total_pixels = image_data.size

    # Create histogram
    plt.hist(image_data.ravel(), bins=256, range=(0, 256), density=False, color='black')

    # Convert y-axis to percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y / total_pixels) * 100:.1f}%'))

    # Set titles and labels

    # Set titles and labels
    plt.title('Histogram of Grayscale Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Percentage')

    # Show the plot
    plt.show()


def find_bin_for_percentage(image: Image.Image, target_percentage: float, ignore_color: Union[List[int], int] = None):
    # Convert image to numpy array
    # noinspection PyTypeChecker
    image_data = np.array(image.convert('L'))

    # Filter out specified color if needed
    if ignore_color is not None:
        image_data = image_data[image_data != ignore_color]

    # Create histogram with raw counts (256 bins for 8-bit grayscale)
    counts, bin_edges = np.histogram(image_data.ravel(), bins=256, range=(0, 256))

    # Calculate cumulative counts and normalize to get cumulative percentage
    cumulative_counts = np.cumsum(counts)
    cumulative_percentage = (cumulative_counts / cumulative_counts[-1]) * 100

    # Find the bin where cumulative percentage meets or exceeds the target
    bin_index = np.searchsorted(cumulative_percentage, target_percentage)
    bin_value = bin_edges[bin_index] if bin_index < len(bin_edges) else None

    return bin_value, cumulative_percentage[bin_index] if bin_value is not None else None


def threshold(img: Image.Image, value=128) -> Image.Image:
    # Open the image and convert it to grayscale
    image = img.convert("L")
    # Apply threshold
    thresholded_image = image.point(lambda p: 255 if p > value else 0)
    return thresholded_image


def levels(img: Image.Image, low=0, high=255, gamma=1.0):
    # Load image and convert to grayscale
    image = img.convert("L")
    # noinspection PyTypeChecker
    image_array = np.array(image, dtype=np.float32)

    # Step 1: Clip the image array to the low and high input levels
    image_array = np.clip((image_array - low) / (high - low), 0, 1)

    # Step 2: Apply gamma correction to adjust midtones
    image_array = np.power(image_array, gamma)

    # Step 3: Scale back to [0, 255] output range
    image_array = (image_array * 255).astype(np.uint8)

    # Convert back to a PIL image
    adjusted_image = Image.fromarray(image_array)
    return adjusted_image


def get_canny_image(
        img,
        low_threshold: float = 100.,
        high_threshold: float = 200.
) -> PIL.Image.Image:
    import cv2
    import numpy as np

    # noinspection PyTypeChecker
    image = np.array(img)
    # noinspection PyUnresolvedReferences
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = PIL.Image.fromarray(image)
    return canny_image


def invert_image(image: np.array) -> np.ndarray:
    if image.dtype == bool:
        return np.bitwise_not(np.array(image, dtype=bool))
    return np.bitwise_not(np.array(image / 255, dtype=bool)).astype(np.uint8) * 255


def get_canny_inverted(image: Union[Image.Image, np.ndarray], low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    canny_image = get_canny_image(image, low_threshold=low_threshold, high_threshold=high_threshold)
    gray_canny = canny_image.convert("L")
    # noinspection PyTypeChecker
    gray_canny_array = np.array(gray_canny)
    canny_inverted = invert_image(gray_canny_array)
    return canny_inverted


def thicken_edges(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    image = invert_image(image)
    # Define the structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # e.g., 3x3 square kernel
    # Apply dilation to thicken the edges
    # noinspection PyUnresolvedReferences
    thickened_image = cv2.dilate(image, kernel, iterations=iterations)
    return invert_image(thickened_image)


def simple_filter(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    image = invert_image(image)
    # Define the structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # e.g., 3x3 square kernel
    # Apply dilation to thicken the edges
    # noinspection PyUnresolvedReferences
    image = cv2.dilate(image, kernel, iterations=iterations)
    # noinspection PyUnresolvedReferences
    image = cv2.erode(image, kernel, iterations=iterations)
    return invert_image(image)


def remove_small_areas_morphology(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # Convert to numpy array
    image_array = np.array(image)

    # Define a kernel for the morphological operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological opening
    # noinspection PyUnresolvedReferences
    cleaned_image = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)

    # Convert back to a PIL image
    return cleaned_image

def _get_canny(mask: np.ndarray) -> Tuple:
    # Canny edge to obtain outlines
    canny_outline = get_canny_inverted(image=mask)
    canny_outline = thicken_edges(canny_outline, iterations=3)
    canny_outline_mask = canny_outline <= 10

    return canny_outline, canny_outline_mask


def selective_morphological_filter(image_array: np.ndarray, kernel_size: int = 3,
                                   intensity_threshold: int = 127) -> np.ndarray:
    # Apply a threshold to create a mask for potential noise areas
    _, mask = cv2.threshold(image_array, intensity_threshold, 255, cv2.THRESH_BINARY)

    # Define a kernel for selective filtering
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Use morphology to target only small, isolated areas (morphological opening)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Use bitwise operations to preserve only the areas of interest
    cleaned_image = cv2.bitwise_and(image_array, cv2.bitwise_not(opened_mask)) + cv2.bitwise_and(image_array,
                                                                                                 opened_mask)

    return cleaned_image


def median_filter(image_array: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """
    Apply median filtering to reduce noise in a grayscale image.

    Parameters:
    - image_array: np.ndarray
        Grayscale image as a numpy array.
    - filter_size: int
        Size of the filter; must be an odd number (e.g., 3, 5, 7).

    Returns:
    - np.ndarray
        Filtered image as a numpy array.
    """
    # Ensure the filter size is odd
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number.")

    # Apply median filtering
    filtered_image = cv2.medianBlur(image_array, filter_size)

    return filtered_image


def thicker_with_midtones(
        image: Union[np.ndarray, Image.Image],
        mid_tone_value: int = 64,
        thickness: int = 1
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('L')

    # Convert the image to a NumPy array
    # noinspection PyTypeChecker
    image_data = np.array(image).astype(np.uint8)

    # Create binary masks for black lines and gray lines
    black_mask = (image_data == 0)
    gray_mask = (image_data == mid_tone_value)

    # Dilate both masks
    dilated_black_mask = ndi.binary_dilation(black_mask, iterations=thickness).astype(np.uint8) * 255
    dilated_gray_mask = ndi.binary_dilation(gray_mask, iterations=thickness).astype(np.uint8) * mid_tone_value

    # Combine the dilated masks and retain the original gray values
    # Replace dilated gray lines with gray
    result_image_data = np.where(dilated_gray_mask > 0, mid_tone_value, image_data)
    # Replace dilated black lines with black
    result_image_data = np.where(dilated_black_mask > 0, 0, result_image_data)

    return result_image_data.astype(np.uint8)


def mean_filter(image_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply mean (average) filtering to reduce noise in a grayscale image.

    Parameters:
    - image_array: np.ndarray
        Grayscale image as a numpy array.
    - kernel_size: int
        Size of the filter; typically an odd number (e.g., 3, 5, 7).

    Returns:
    - np.ndarray
        Filtered image as a numpy array.
    """
    # Apply mean (average) filtering
    filtered_image = cv2.blur(image_array, (kernel_size, kernel_size))

    return filtered_image


def main():
    breed = "boston_bull"
    base_dir = Path("/mnt/LinuxData/edge/training/") / breed
    sketch_dir, mask_dir = base_dir / "sketch", base_dir / "mask"
    result_dir = Path("/mnt/LinuxData/edge/tests")

    # img_path = "/mnt/LinuxData/edge/training/yorkshire_terrier/sketch/006.png"
    # mask_path = "/mnt/LinuxData/edge/training/yorkshire_terrier/mask/006.png"


    for image_id in range(1, 20):
        # noinspection PyBroadException
        try:
            img_path = sketch_dir / (str(image_id).zfill(3) + ".png")
            mask_path = mask_dir / (str(image_id).zfill(3) + ".png")
            result_path = result_dir / (str(image_id).zfill(3) + ".png")

            sketch_image = Image.open(img_path).convert('RGB')
            mask_image = Image.open(mask_path).convert('L')

            # img_threshold = threshold(sketch_image, value=192)
            # img_threshold.show()

            # histogram(sketch_image, ignore_color=255)

            bin_at_percents = find_bin_for_percentage(
                image=sketch_image, target_percentage=TARGET_PERCENTAGE, ignore_color=255
            )
            low_tone_threshold = bin_at_percents[0]
            print(low_tone_threshold)

            img_levels_adjusted = levels(sketch_image, low=low_tone_threshold, gamma=1.0, high=250)
            # noinspection PyTypeChecker
            img_levels_adjusted_arr = np.array(img_levels_adjusted).astype(np.uint8)
            # img_levels_adjusted.show()

            color_filter_threshold_mid = find_bin_for_percentage(image=img_levels_adjusted, target_percentage=50, ignore_color=255)[0]
            color_filter_threshold_mid_high = find_bin_for_percentage(image=img_levels_adjusted, target_percentage=75, ignore_color=255)[0]

            # noinspection PyTypeChecker
            mask = ~(np.array(mask_image) / 255).astype(bool)

            # noinspection PyTypeChecker
            canny_outline, canny_mask = _get_canny(np.array(mask_image).astype(np.uint8))

            # noinspection PyTypeChecker
            image_arr = np.array(img_levels_adjusted_arr).astype(np.uint8)
            image_arr[mask] = 255

            # ##############
            thresh_low_tones = image_arr < color_filter_threshold_mid
            thresh_high_tones = image_arr >= color_filter_threshold_mid
            thresh_mid_tones = (image_arr >= color_filter_threshold_mid) & (image_arr < color_filter_threshold_mid_high)
            image_arr[thresh_low_tones] = COLOR_FILTER_VALUE_LOW
            image_arr[thresh_high_tones] = COLOR_FILTER_VALUE_HIGH
            image_arr[thresh_mid_tones] = COLOR_FILTER_VALUE_MID
            # ##############

            # image_arr = median_filter(image_arr, filter_size=3)
            # image_arr = mean_filter(image_arr, kernel_size=3)

            canny_mask = ~thicken_edges((~canny_mask).astype(np.uint8) * 255, iterations=1).astype(bool)
            canny_outline = thicken_edges(canny_outline, iterations=1)
            image_arr[canny_mask] = canny_outline[canny_mask]

            # Image.fromarray(image_arr.astype(np.uint8)).convert('L').show()
            # ##############
            # noinspection PyTypeChecker
            # merged_image_arr = np.array(image_arr).copy()
            # merged_image_arr = thicker_with_midtones(merged_image_arr, COLOR_FILTER_VALUE_MID)
            # merged_image_arr[thresh_mid_tones] = COLOR_FILTER_VALUE_MID
            # merged_image_arr = median_filter(merged_image_arr)
            # merged_image_arr[canny_mask] = canny_outline[canny_mask]
            # ##############


            # image_arr = median_filter(image_arr, filter_size=3)
            result = Image.fromarray(image_arr.astype(np.uint8)).convert('L')

            # break
            result.save(result_path)
        except Exception:
            continue


if __name__ == '__main__':
    main()

