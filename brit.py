import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# noinspection PyUnresolvedReferences
def load_rgb(path: str) -> np.ndarray:
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


def transform(img: np.ndarray, tranformation_vec: np.ndarray):
    r, g, b, a = img.reshape(-1, 4).T
    uint32_img = r | (g << 8) | (b << 16) | (a << 24)

    u = 0
    while 256 << u < len(tranformation_vec):
        u += 1

    t = time.process_time()

    r_transform = (uint32_img >> 0 & 255) << u
    g_transform = (uint32_img >> 8 & 255) << u
    b_transform = (uint32_img >> 16 & 255) << u

    r_tv = tranformation_vec[r_transform].astype(np.uint32)
    g_tv = tranformation_vec[g_transform].astype(np.uint32)
    b_tv = tranformation_vec[b_transform].astype(np.uint32)

    return r_tv | (g_tv << 8) | (b_tv << 16) | uint32_img & 4278190080


def uint32_to_rgb(uint32_img: np.ndarray, width: int, height: int) -> np.ndarray:
    r = np.uint8(uint32_img)
    g = np.uint8(uint32_img >> 8)
    b = np.uint8(uint32_img >> 16)
    a = np.uint8(uint32_img >> 24)
    return np.stack([r, g, b, a]).T.reshape(height, width, 4)


def value_rescaler(value, hrzn, vrtc, buffer):
    # Clip.
    if value <= hrzn[0]:
        return vrtc[0]
    if value >= hrzn[len(hrzn) - 1]:
        return vrtc[len(hrzn) - 1]

    # Find interval.
    right = 1
    while hrzn[right] < value:
        right += 1
    left = right - 1

    *_, buffer_row = buffer[left]

    hrzn_interval = [hrzn[left], hrzn[right]]
    vrtc_interval = [vrtc[left], vrtc[right]]

    hrzn_length = hrzn_interval[1] - hrzn_interval[0]
    vrtc_length = vrtc_interval[1] - vrtc_interval[0]

    scale = (value - hrzn_interval[0]) / hrzn_length

    low = buffer_row[left] * hrzn_length - vrtc_length
    high = - buffer_row[right] * hrzn_length + vrtc_length

    f = (1 - scale) * vrtc_interval[0] + scale * vrtc_interval[1] + \
        scale * (1 - scale) * (low * (1 - scale) + high * scale)
    return f


def get_buffer(matrix, buffer):
    nrows = len(matrix)

    for row_index in range(nrows):
        i = 0
        comparator = -np.inf

        for p_index in range(row_index, nrows):
            if np.abs(matrix[p_index][row_index]) > comparator:
                i = p_index
                comparator = np.abs(matrix[p_index][row_index])

        matrix[row_index], matrix[i] = matrix[i], matrix[row_index]

        for pp_index in range(row_index + 1, nrows):
            if matrix[row_index][row_index] == 0:
                return buffer

            n = matrix[pp_index][row_index] / matrix[row_index][row_index]
            for q_index in range(row_index, nrows + 1):
                matrix[pp_index][q_index] -= matrix[row_index][q_index] * n

    for index in range(nrows - 1, -1, -1):
        if matrix[index][index] == 0:
            return buffer

        z = matrix[index][nrows] / matrix[index][index]
        buffer[index] = z

        for q_index in range(index - 1, -1, -1):
            matrix[q_index][nrows] -= matrix[q_index][index] * z
            matrix[q_index][index] = 0
    return buffer


def magic(xs, ys, buffer):
    i = len(xs) - 1
    matrix = np.zeros((i + 1) * (i + 2)).reshape(i + 1, i + 2)

    matrix[0][0] = 2 / (xs[1] - xs[0])
    matrix[0][1] = 1 / (xs[1] - xs[0])
    matrix[0][i + 1] = 3 * (ys[1] - ys[0]) / ((xs[1] - xs[0]) * (xs[1] - xs[0]))
    matrix[i][i - 1] = 1 / (xs[i] - xs[i - 1])
    matrix[i][i] = 2 / (xs[i] - xs[i - 1])
    matrix[i][i + 1] = 3 * (ys[i] - ys[i - 1]) / ((xs[i] - xs[i - 1]) * (xs[i] - xs[i - 1]))

    for idx in range(1, i):
        matrix[idx][idx - 1] = 1 / (xs[idx] - xs[idx - 1])
        matrix[idx][idx] = 2 * (1 / (xs[idx] - xs[idx - 1]) + 1 / (xs[idx + 1] - xs[idx]))
        matrix[idx][idx + 1] = 1 / (xs[idx + 1] - xs[idx])
        matrix[idx][i + 1] = 3 * ((ys[idx] - ys[idx - 1]) / ((xs[idx] - xs[idx - 1]) * (xs[idx] - xs[idx - 1]))
                                  + (ys[idx + 1] - ys[idx]) / ((xs[idx + 1] - xs[idx]) * (xs[idx + 1] - xs[idx])))

    get_buffer(matrix, buffer)
    return buffer


def adjustment_vector_estimator(hrzn, vrtc, transformer_length):
    min_, max_ = 0, 255
    hrzn, vrtc = hrzn[:], vrtc[:]
    y = [hrzn, vrtc, [*np.zeros(len(hrzn))]]
    t_buffer = [y] * 4
    magic(*y)

    adjustment_vector = [*np.zeros(transformer_length)]
    for X in range(transformer_length):
        j = value_rescaler(X * (255 / (transformer_length - 1)), hrzn, vrtc, t_buffer)
        adjustment_vector[X] = 1 / 255 * np.max([min_, np.min([max_, j])])
    return adjustment_vector


def get_transformator(contrast_, brightness_):
    adjustment_vector_size = 1024

    def for_contrast_():
        contrast_shift = -30 + 60 * (contrast_ + 100) / 200
        r = [[i / 3 * 255, i / 3 * 255] for i in range(4)]
        r[1][0] = 64
        r[1][1] = 64 - contrast_shift
        r[2][0] = 128 + 64
        r[2][1] = 128 + 64 + contrast_shift

        sorted(r, key=lambda v: v[0])
        hrzn, vrtc = list(zip(*r))
        return adjustment_vector_estimator(hrzn, vrtc, adjustment_vector_size)

    def for_brightness_():
        brightness_shift = np.abs(brightness_) / 100
        rb = [[i / 3 * 255, i / 3 * 255] for i in range(4)]
        rb[1][0] = 130 - brightness_shift * 26
        rb[1][1] = 130 + brightness_shift * 51
        rb[2][0] = 233 - brightness_shift * 48
        rb[2][1] = 233 + brightness_shift * 10
        hrzn, vrtc = list(zip(*rb))
        vector = adjustment_vector_estimator(hrzn, vrtc, adjustment_vector_size)

        # Adjust values for brightness lower then 0.
        if brightness_ < 0:
            x = [*np.zeros(adjustment_vector_size)]
            inverse_size = 1 / adjustment_vector_size

            for index in range(adjustment_vector_size):
                inverted_index = index * inverse_size
                i = index
                while vector[i] > inverted_index and i > 1:
                    i -= 1
                x[index] = i * inverse_size
            vector = x

        return vector

    contrast_adjustment = for_contrast_()
    brightness_adjustment = for_brightness_()

    adjustment_vector = np.zeros(adjustment_vector_size, dtype=np.uint8)
    for idx in range(adjustment_vector_size):
        ctr_index = int(np.round((adjustment_vector_size - 1) * brightness_adjustment[idx]))
        adjustment_vector[idx] = np.round(255 * contrast_adjustment[ctr_index])

    return adjustment_vector


def main():
    """
    h x w x 4
    :return:
    """
    path = "/home/mike/Projects/photoshop/banner-4-medium.jpg"
    rgb_orig = load_rgb(path).astype(np.uint8).astype(np.uint32)

    transformer_vec = get_transformator(contrast_=0, brightness_=20)

    t = time.process_time()
    uint32 = transform(rgb_orig, transformer_vec)
    print('Total: ', time.process_time() - t)

    rgb_brit = uint32_to_rgb(uint32, *rgb_orig.shape[:2][::-1])

    fix, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(rgb_orig)
    axs[1].imshow(rgb_brit)
    plt.show()


if __name__ == '__main__':
    main()
