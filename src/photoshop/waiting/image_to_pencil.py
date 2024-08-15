from typing import Union

import cv2
import np
from PIL import Image


def image_to_pencil_sketch(image: Union[Image, str, np.ndarray]):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    image = image.convert('RGB')

    # noinspection PyTypeChecker
    image_array = np.array(image)

    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(invert, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray_image, inverted_blur, scale=256.)
    return sketch