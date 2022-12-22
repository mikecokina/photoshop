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


def magic(hrzn, vrtc, buffer):
    i = len(hrzn) - 1
    matrix = np.zeros((i + 1) * (i + 2)).reshape(i + 1, i + 2)

    matrix[0][0] = 2 / (hrzn[1] - hrzn[0])
    matrix[0][1] = 1 / (hrzn[1] - hrzn[0])
    matrix[0][i + 1] = 3 * (vrtc[1] - vrtc[0]) / ((hrzn[1] - hrzn[0]) * (hrzn[1] - hrzn[0]))
    matrix[i][i - 1] = 1 / (hrzn[i] - hrzn[i - 1])
    matrix[i][i] = 2 / (hrzn[i] - hrzn[i - 1])
    matrix[i][i + 1] = 3 * (vrtc[i] - vrtc[i - 1]) / ((hrzn[i] - hrzn[i - 1]) * (hrzn[i] - hrzn[i - 1]))

    for idx in range(1, i):
        matrix[idx][idx - 1] = 1 / (hrzn[idx] - hrzn[idx - 1])
        matrix[idx][idx] = 2 * (1 / (hrzn[idx] - hrzn[idx - 1]) + 1 / (hrzn[idx + 1] - hrzn[idx]))
        matrix[idx][idx + 1] = 1 / (hrzn[idx + 1] - hrzn[idx])
        matrix[idx][i + 1] = 3 * (
                    (vrtc[idx] - vrtc[idx - 1]) / ((hrzn[idx] - hrzn[idx - 1]) * (hrzn[idx] - hrzn[idx - 1]))
                    + (vrtc[idx + 1] - vrtc[idx]) / ((hrzn[idx + 1] - hrzn[idx]) * (hrzn[idx + 1] - hrzn[idx])))

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


def get_transformator(brightness_, contrast_):
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


def J_kK_multiply(w, n):
    # return np.array(n) * np.array(w)

    m = np.zeros(16)
    m[0] = w[0] * n[0] + w[1] * n[4] + w[2] * n[8] + w[3] * n[12]
    m[1] = w[0] * n[1] + w[1] * n[5] + w[2] * n[9] + w[3] * n[13]
    m[2] = w[0] * n[2] + w[1] * n[6] + w[2] * n[10] + w[3] * n[14]
    m[3] = w[0] * n[3] + w[1] * n[7] + w[2] * n[11] + w[3] * n[15]
    m[4] = w[4] * n[0] + w[5] * n[4] + w[6] * n[8] + w[7] * n[12]
    m[5] = w[4] * n[1] + w[5] * n[5] + w[6] * n[9] + w[7] * n[13]
    m[6] = w[4] * n[2] + w[5] * n[6] + w[6] * n[10] + w[7] * n[14]
    m[7] = w[4] * n[3] + w[5] * n[7] + w[6] * n[11] + w[7] * n[15]
    m[8] = w[8] * n[0] + w[9] * n[4] + w[10] * n[8] + w[11] * n[12]
    m[9] = w[8] * n[1] + w[9] * n[5] + w[10] * n[9] + w[11] * n[13]
    m[10] = w[8] * n[2] + w[9] * n[6] + w[10] * n[10] + w[11] * n[14]
    m[11] = w[8] * n[3] + w[9] * n[7] + w[10] * n[11] + w[11] * n[15]
    m[12] = w[12] * n[0] + w[13] * n[4] + w[14] * n[8] + w[15] * n[12]
    m[13] = w[12] * n[1] + w[13] * n[5] + w[14] * n[9] + w[15] * n[13]
    m[14] = w[12] * n[2] + w[13] * n[6] + w[14] * n[10] + w[15] * n[14]
    m[15] = w[12] * n[3] + w[13] * n[7] + w[14] * n[11] + w[15] * n[15]
    return m


def J_kK_hY(w):
    n = ~~(int(w + 0.5))
    return 0 if n < 0 else 255 if n > 255 else n


def J_kK_transform(w, n, m):
    M = J_kK_hY
    y = len(w['p'])

    for X in range(y):
        T = w['p'][X]
        N = w['w'][X]
        Q = w['k'][X]
        n['p'][X] = M(m[0] * T + m[1] * N + m[2] * Q + m[3] * 255)
        n['w'][X] = M(m[4] * T + m[5] * N + m[6] * Q + m[7] * 255)
        n['k'][X] = M(m[8] * T + m[9] * N + m[10] * Q + m[11] * 255)


def get_legacy_transformer(brightness_, contrast_):
    """
    var Q = brightness_ / 255, Z = 1 + contrast_ / 100;
    if (Z > 1) Z = 1 + Math.tan(Math.PI / 2 * contrast_ / 101);
    var h = (1 - Z) / 2, j = J.kK.uw(Q, Q, Q), f = [Z, 0, 0, h, 0, Z, 0, h, 0, 0, Z, h, 0, 0, 0, 1],
        D = J.kK.multiply(f, j), Y = new J.UH(256);
    for (var X = 0; X < 256; X++) Y.p[X] = X;
    J.kK.transform(Y, Y, D);
    """
    Q = brightness_ / 255
    Z = 1 + contrast_ / 100

    if Z > 1:
        Z = 1 + np.tan(np.pi / 2 * contrast_ / 101)

    h = (1 - Z) / 2
    j = [1, 0, 0, Q, 0, 1, 0, Q, 0, 0, 1, Q, 0, 0, 0, 1]
    f = [Z, 0, 0, h, 0, Z, 0, h, 0, 0, Z, h, 0, 0, 0, 1]

    D = J_kK_multiply(f, j)
    Y = {
        'G1': np.zeros(256, dtype=np.uint8),
        'p': np.arange(256, dtype=np.uint8),
        'w': np.zeros(256, dtype=np.uint8),
        'k': np.zeros(256, dtype=np.uint8)
    }

    J_kK_transform(Y, Y, D)

    return Y['p']


def britcntr(rgba, brightnes_, contrast_):
    img = rgba.copy()
    img = img + brightnes_
    img = np.clip(img, 0, 255)

    old_scale_min = -100.0
    old_scale_max = 100.0
    new_scale_min = -255.0
    new_scale_max = 255.0

    old_scale_range = old_scale_max - old_scale_min
    new_scale_range = new_scale_max - new_scale_min

    # 0 to 129.5
    contrast = ((contrast_ - old_scale_min) * new_scale_range / old_scale_range) + new_scale_min
    factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))

    img = factor * (img - 128) + 128
    img = np.clip(img, 0, 255)

    return img


def main():
    path = "/home/mike/Projects/photoshop/banner-4-medium.jpg"
    rgb_orig = load_rgb(path).astype(np.uint32)

    brit, cntr = 0, 100

    transformer_vec = get_legacy_transformer(brightness_=brit, contrast_=cntr)
    uint32 = transform(rgb_orig, transformer_vec)
    rgb_brit = uint32_to_rgb(uint32, *rgb_orig.shape[:2][::-1])

    rgb_std = britcntr(rgb_orig, brightnes_=brit, contrast_=cntr)

    fix, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(rgb_orig)
    axs[1].imshow(rgb_brit)
    axs[2].imshow(rgb_std)
    plt.show()





    # from photoshop import brightness
    # from photoshop._io import load_rgba
    #
    # path = "/home/mike/Projects/photoshop/banner-4-medium.jpg"
    # rgb_orig = load_rgb(path).astype(np.uint8)
    #
    # img = brightness(rgb_orig, brightness=50)
    # fix, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].imshow(rgb_orig)
    # axs[1].imshow(img)
    # plt.show()

    #
    # # """
    # # h x w x 4
    # # :return:
    # # """
    # path = "/home/mike/Projects/photoshop/banner-4-medium.jpg"
    # rgb_orig = load_rgb(path).astype(np.uint8).astype(np.uint32)
    #
    # transformer_vec = get_transformator(contrast_=0, brightness_=20)
    #
    # t = time.process_time()
    # uint32 = transform(rgb_orig, transformer_vec)
    # print('Total: ', time.process_time() - t)
    #
    # rgb_brit = uint32_to_rgb(uint32, *rgb_orig.shape[:2][::-1])
    #
    # fix, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].imshow(rgb_orig)
    # axs[1].imshow(rgb_brit)
    # plt.show()


if __name__ == '__main__':
    main()
