"""
function whatever(w, n, T, C, a) {
    // w: Uint8Array of 5000 length
    // n: Uint8Array of 5000 length
    // C = T = a: Uint8Array of 1024 length

    let f = new Uint32Array(w.buffer),
        O = new Uint32Array(n.buffer),
        d = f.length,
        u = 0;
    while (256 << u < T.length)
        u++;
    for (let X = 0; X !== d; X++) {
        let e = f[X],
            F = (e & 255) << u,
            c = (e >>> 8 & 255) << u,
            k = (e >>> 16 & 255) << u,
            z = T[F],
            h = C[c],
            g = a[k];
        O[X] = z | h << 8 | g << 16 | e & 4278190080
    }
}
"""
from pprint import pprint
from typing import Dict, List

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
    uint32_new = uint32_img.copy()
    u = 0

    while 256 << u < len(tranformation_vec):
        u += 1

    for pixel_index in range(len(uint32_img)):
        pixel = uint32_img[pixel_index]

        r_transform = (pixel >> 0 & 255) << u
        g_transform = (pixel >> 8 & 255) << u
        b_transform = (pixel >> 16 & 255) << u

        r_tv = tranformation_vec[r_transform]
        g_tv = tranformation_vec[g_transform]
        b_tv = tranformation_vec[b_transform]

        uint32_new[pixel_index] = r_tv | (g_tv << 8) | (b_tv << 16) | pixel & 4278190080

    return uint32_new


def uint32_to_rgb(uint32_img: np.ndarray, width: int, height: int) -> np.ndarray:
    r = np.uint8(uint32_img)
    g = np.uint8(uint32_img >> 8)
    b = np.uint8(uint32_img >> 16)
    a = np.uint8(uint32_img >> 24)
    return np.stack([r, g, b, a]).T.reshape(height, width, 4)


def J_gZ_azc(w: Dict):
    n = {'Ev': [], 'bH': [], 'QA': []}
    for X in range(len(w)):
        n['Ev'].append(w[X]['Hrzn'])
        n['bH'].append(w[X]['Vrtc'])
        n['QA'].append(True)

    return n


def J_gZ_aqu(w, n, m, M):
    if w <= n[0]:
        return m[0]
    if w >= n[len(n) - 1]:
        return m[len(n) - 1]

    X = 1
    while n[X] < w:
        X += 1

    y = n[X]
    T = n[X - 1]
    N = m[X]
    Q = m[X - 1]
    Z = (w - T) / (y - T)
    h = M[X - 1] * (y - T) - (N - Q)
    j = -M[X] * (y - T) + (N - Q)
    f = (1 - Z) * Q + Z * N + Z * (1 - Z) * (h * (1 - Z) + j * Z)
    return f


def J_gZ_azJ(w, n, m, M):
    if w <= n[0]:
        return m[0]
    if w >= n[len(n) - 1]:
        return m[len(n) - 1]
    X = 1
    while n[X] < w:
        X += 1
    y = M[X - 1]
    return J_gZ_aqu(w, y['Ev'], y['bH'], y['M0'])


def J_Cu_a1r(w, n, m):
    M = w[n]
    w[n] = w[m]
    w[m] = M


def J_Cu_ZI(w, n):
    m = len(w)

    for M in range(m):
        y = 0
        T = -np.inf

        for X in range(M, m):
            if np.abs(w[X][M]) > T:
                y = X
                T = np.abs(w[X][M])

        J_Cu_a1r(w, M, y)
        for X in range(M + 1, m):
            if w[M][M] == 0:
                return 1
            N = w[X][M] / w[M][M]
            for Q in range(M, m + 1):
                w[X][Q] -= w[M][Q] * N

    for X in range(m - 1, -1, -1):
        if w[X][X] == 0:
            return 1
        Z = w[X][m] / w[X][X]
        n[X] = Z

        for Q in range(X - 1, -1, -1):
            w[Q][m] -= w[Q][X] * Z
            w[Q][X] = 0
    return 0


def J_Cu_Cd(w, n):
    m = []
    for X in range(w):
        m.append([])
        for M in range(n):
            m[X].append(0.0)
    return m


def J_gZ_a3x(w, n, m):
    ak = len(w) - 1
    M = J_Cu_Cd(ak + 1, ak + 2)

    for X in range(1, ak):

        M[X][X - 1] = 1 / (w[X] - w[X - 1])
        M[X][X] = 2 * (1 / (w[X] - w[X - 1]) + 1 / (w[X + 1] - w[X]))
        M[X][X + 1] = 1 / (w[X + 1] - w[X])
        M[X][ak + 1] = 3 * ((n[X] - n[X - 1]) / ((w[X] - w[X - 1]) * (w[X] - w[X - 1])) + (n[X + 1] - n[X]) / (
                    (w[X + 1] - w[X]) * (w[X + 1] - w[X])))

    M[0][0] = 2 / (w[1] - w[0])
    M[0][1] = 1 / (w[1] - w[0])
    M[0][ak + 1] = 3 * (n[1] - n[0]) / ((w[1] - w[0]) * (w[1] - w[0]))
    M[ak][ak - 1] = 1 / (w[ak] - w[ak - 1])
    M[ak][ak] = 2 / (w[ak] - w[ak - 1])
    M[ak][ak + 1] = 3 * (n[ak] - n[ak - 1]) / ((w[ak] - w[ak - 1]) * (w[ak] - w[ak - 1]))

    J_Cu_ZI(M, m)


def J_gZ_a6F(Ev, bH, QA, M):
    M0 = [*np.zeros(len(Ev))]
    y = {'Ev': [Ev[0]], 'bH': [bH[0]], 'M0': M0}
    M[0] = y

    for X in range(1, len(Ev) - 1):
        y['Ev'].append(Ev[X])
        y['bH'].append(bH[X])

        # Never satisfied for contrast
        # if QA[X] is False:
        #     J_gZ_a3x(y['Ev'], y['bH'], y['M0'])
        #     y = {'Ev': [Ev[X]], 'bH': [bH[X]], 'M0': []}

        M[X] = y

    y['Ev'].append(Ev[-1])
    y['bH'].append(bH[-1])
    J_gZ_a3x(y['Ev'], y['bH'], y['M0'])
    M[-1] = y


def J_gZ_YO(w, transformer_length, is_float=False):
    min_, max_ = 0, 255
    if is_float:
        min_, max_ = -1e9, 1e9

    y = J_gZ_azc(w)
    T = [*np.zeros(len(y['Ev']))]

    J_gZ_a6F(y['Ev'], y['bH'], y['QA'], T)

    N = [*np.zeros(transformer_length)]
    for X in range(transformer_length):
        J = J_gZ_azJ(X * (255 / (transformer_length - 1)), y['Ev'], y['bH'], T)
        N[X] = 1 / 255 * np.max([min_, np.min([max_, J])])
    return N


def J_gZ_ya(w, n, m):
    return {
        'Hrzn': w,
        'Vrtc': n,
        'Cnty': m
    }


def get_transformator(contrast_, brightness_):
    H = 1024
    unknown_quantity = -30 + 60 * (contrast_ + 100) / 200

    R = [{'Hrzn': i / 3 * 255, 'Vrtc': i / 3 * 255, 'Cnty': True} for i in range(4)]
    R[1]['Hrzn'] = 64
    R[1]['Vrtc'] = 64 - unknown_quantity
    R[2]['Hrzn'] = 128 + 64
    R[2]['Vrtc'] = 128 + 64 + unknown_quantity

    sorted(R, key=lambda x: x['Hrzn'])
    g = J_gZ_YO(R, H)

    def for_brightness_(jE, H):
        R = []
        ak = 3
        for X in range(ak + 1):
            R.append(J_gZ_ya(X / ak * 255, X / ak * 255, True))

        R[1]['Hrzn'] = 130 - jE * 26
        R[1]['Vrtc'] = 130 + jE * 51
        R[2]['Hrzn'] = 233 - jE * 48
        R[2]['Vrtc'] = 233 + jE * 10
        return J_gZ_YO(R, H)

    W = for_brightness_(np.abs(brightness_) / 100, H)

    if brightness_ < 0:
        x = [*np.zeros(H)]
        S = 1 / H

        for X in range(H):
            C = X * S
            c = X
            while W[c] > C and c > 1:
                c -= 1
            x[X] = c * S
        W = x

    p = np.zeros(H, dtype=np.uint8)

    for X in range(H):
        a = int(np.round((H - 1) * W[X]))
        p[X] = np.round(255 * g[a])


def main():
    transformer_vec = np.array([
        0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11,
        12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 19, 19, 19, 20, 20, 20, 21,
        21, 22, 22, 22, 23, 23, 23, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30,
        31, 31, 31, 32, 32, 32, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 38, 38, 38, 39, 39, 39,
        40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 48, 48, 48, 49,
        49, 49, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 58, 58,
        58, 59, 59, 59, 60, 60, 60, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 66, 66, 66, 67, 67,
        67, 68, 68, 68, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76,
        76, 77, 77, 77, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 84, 84, 84, 85, 85,
        85, 85, 86, 86, 86, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 94,
        94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 101, 101, 101, 102,
        102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108,
        109, 109, 109, 110, 110, 110, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115,
        116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122,
        122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129,
        129, 129, 130, 130, 130, 131, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135,
        135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 140, 141, 141, 141,
        142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 146, 147, 147, 147, 148,
        148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154,
        154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160,
        160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 163, 164, 164, 164, 165, 165, 165, 165, 166,
        166, 166, 167, 167, 167, 168, 168, 168, 168, 169, 169, 169, 170, 170, 170, 170, 170, 171, 171, 171,
        172, 172, 172, 172, 173, 173, 173, 174, 174, 174, 174, 175, 175, 175, 175, 176, 176, 176, 177, 177,
        177, 177, 178, 178, 178, 179, 179, 179, 179, 180, 180, 180, 180, 181, 181, 181, 182, 182, 182, 182,
        183, 183, 183, 183, 184, 184, 184, 184, 185, 185, 185, 186, 186, 186, 186, 187, 187, 187, 187, 188,
        188, 188, 188, 189, 189, 189, 189, 190, 190, 190, 190, 191, 191, 191, 191, 192, 192, 192, 192, 193,
        193, 193, 193, 194, 194, 194, 194, 195, 195, 195, 195, 196, 196, 196, 196, 197, 197, 197, 197, 198,
        198, 198, 198, 199, 199, 199, 199, 200, 200, 200, 200, 201, 201, 201, 201, 201, 202, 202, 202, 202,
        203, 203, 203, 203, 204, 204, 204, 204, 205, 205, 205, 205, 205, 206, 206, 206, 206, 207, 207, 207,
        207, 207, 208, 208, 208, 208, 209, 209, 209, 209, 209, 210, 210, 210, 210, 211, 211, 211, 211, 211,
        212, 212, 212, 212, 212, 213, 213, 213, 213, 214, 214, 214, 214, 214, 215, 215, 215, 215, 215, 216,
        216, 216, 216, 216, 217, 217, 217, 217, 217, 218, 218, 218, 218, 218, 219, 219, 219, 219, 219, 220,
        220, 220, 220, 220, 221, 221, 221, 221, 221, 221, 222, 222, 222, 222, 222, 223, 223, 223, 223, 223,
        223, 224, 224, 224, 224, 224, 225, 225, 225, 225, 225, 225, 226, 226, 226, 226, 226, 227, 227, 227,
        227, 227, 227, 228, 228, 228, 228, 228, 228, 229, 229, 229, 229, 229, 229, 230, 230, 230, 230, 230,
        230, 230, 231, 231, 231, 231, 231, 231, 232, 232, 232, 232, 232, 232, 233, 233, 233, 233, 233, 233,
        233, 234, 234, 234, 234, 234, 234, 234, 235, 235, 235, 235, 235, 235, 235, 236, 236, 236, 236, 236,
        236, 236, 237, 237, 237, 237, 237, 237, 237, 237, 238, 238, 238, 238, 238, 238, 238, 238, 239, 239,
        239, 239, 239, 239, 239, 239, 240, 240, 240, 240, 240, 240, 240, 240, 241, 241, 241, 241, 241, 241,
        241, 241, 241, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 243, 243, 243, 243, 243, 243, 243,
        243, 243, 243, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 245, 245, 245, 245, 245, 245, 245,
        245, 245, 245, 245, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 247, 247, 247, 247, 247,
        247, 247, 247, 247, 247, 247, 247, 247, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248,
        249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251,
        251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
        253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254,
        254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255
    ], dtype=np.uint8)
    """
    h x w x 4
    :return:
    """
    # path = "/home/mike/Projects/photoshop/banner-4.jpg"
    # rgb_orig = load_rgb(path).astype(np.uint8).astype(np.uint32)

    get_transformator(contrast_=0, brightness_=-20)

    # Brightness
    # uint32 = transform(rgb_orig, transformer_vec)
    # rgb_brit = uint32_to_rgb(uint32, *rgb_orig.shape[:2][::-1])
    #
    # fix, axs = plt.subplots(nrows=1, ncols=2)
    #
    # axs[0].imshow(rgb_orig)
    # axs[1].imshow(rgb_brit)
    # plt.show()

    # img = np.array(
    #     [
    #         [
    #             [0, 1, 0, 255], [1, 2, 0, 255], [1, 3, 0, 255]
    #         ],
    #         [
    #             [255, 255, 255, 255], [255, 255, 255, 255], [255, 255, 255, 255]
    #         ],
    #         [
    #             [255, 255, 255, 255], [255, 255, 255, 255], [255, 255, 255, 255]
    #         ],
    #         [
    #             [255, 255, 255, 255], [255, 255, 255, 255], [255, 255, 255, 255]
    #         ],
    #         [
    #             [254, 254, 254, 254], [253, 253, 253, 253], [251, 251, 251, 251]
    #         ]
    #     ]
    #     , dtype=np.uint32)
    #
    # vec = np.ones(1024, dtype=np.uint8)
    # uint32 = transform(img, vec)
    # uint32_to_rgb(uint32, *img.shape[:2][::-1])
    # print(uint32)


if __name__ == '__main__':
    main()
