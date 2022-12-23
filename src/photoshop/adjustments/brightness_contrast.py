from abc import abstractmethod
from typing import Union, Tuple

from photoshop._interface import Command
from photoshop.core.error import ValidationError
from photoshop.libs.numpy import np
from photoshop.core.dtype import UInt8, Int, Float, UInt32, Bool
from photoshop.ops.transform import uint32_to_rgba


class AbstractPhotopeaReceiver(object):
    @classmethod
    def transform(cls, rgba: UInt8, tranformation_vector: np.ndarray) -> Union[UInt8, np.ndarray]:
        """
        Transform original rgba image to new adjusted by brightness_contrast transformation vector.

        :param rgba: UInt8;
        :param tranformation_vector: np.ndarray;
        :return: Union[UInt8, np.ndarray];
        """
        height, width, *_ = rgba.shape
        r, g, b, a = np.array(rgba.reshape(-1, 4).T, dtype=UInt32)
        uint32_img = r | (g << 8) | (b << 16) | (a << 24)
        del r, g, b, a

        u = 0
        while 256 << u < len(tranformation_vector):
            u += 1

        r_transform = (uint32_img >> 0 & 255) << u
        g_transform = (uint32_img >> 8 & 255) << u
        b_transform = (uint32_img >> 16 & 255) << u

        r_tv = tranformation_vector[r_transform].astype(UInt32)
        g_tv = tranformation_vector[g_transform].astype(UInt32)
        b_tv = tranformation_vector[b_transform].astype(UInt32)

        del r_transform, g_transform, b_transform

        uint32_new = r_tv | (g_tv << 8) | (b_tv << 16) | uint32_img & 4278190080

        del r_tv, g_tv, b_tv

        return uint32_to_rgba(uint32_new, width=width, height=height)

    @classmethod
    def transformation_vector(cls, brit: Int, cntr: Int) -> Union[UInt8, np.ndarray]:
        """
        Compute transformation vector for given brightness and contrast.

        :param brit: Int; brightness
        :param cntr: Int; contrast
        :return: Union[UInt8, np.ndarray];
        """
        return cls.get_transformator(brightness=brit, contrast=cntr)

    @staticmethod
    @abstractmethod
    def get_transformator(brightness: Int, contrast: Int) -> Union[UInt8, np.ndarray]:
        pass


class PhotopeaReceiver(AbstractPhotopeaReceiver):
    @staticmethod
    def value_rescaler(value: Float, hrzn: Tuple, vrtc: Tuple, buffer: np.ndarray) -> Float:
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

    @staticmethod
    def get_buffer(matrix: np.ndarray, buffer: np.ndarray) -> np.ndarray:
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

    @classmethod
    def magic(cls, hrzn: np.ndarray, vrtc: np.ndarray, buffer: np.ndarray) -> np.ndarray:
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

        cls.get_buffer(matrix, buffer)
        return buffer

    @classmethod
    def adjustment_vector_estimator(cls, hrzn: Tuple, vrtc: Tuple, transformer_length: Int) -> np.ndarray:
        min_, max_ = 0, 255
        hrzn, vrtc = hrzn[:], vrtc[:]
        y = np.array([hrzn, vrtc, [*np.zeros(len(hrzn))]], dtype=Float)
        t_buffer = np.array([y] * 4, dtype=Float)
        cls.magic(*y)

        adjustment_vector = [*np.zeros(transformer_length)]
        for X in range(transformer_length):
            j = cls.value_rescaler(X * (255 / (transformer_length - 1)), hrzn, vrtc, t_buffer)
            adjustment_vector[X] = 1 / 255 * np.max([min_, np.min([max_, j])])
        return np.array(adjustment_vector, dtype=Float)

    @classmethod
    def get_transformator(cls, brightness: Int, contrast: Int) -> Union[UInt8, np.ndarray]:
        """
        Compute transformation vector based on given brightness/contrast.

        :param brightness: Int;
        :param contrast: Int;
        :return: Union[UInt8, np.ndarray];
        """
        adjustment_vector_size = 1024

        def for_contrast_() -> np.ndarray:
            contrast_shift = -30 + 60 * (contrast + 100) / 200
            r = [[i / 3 * 255, i / 3 * 255] for i in range(4)]
            r[1][0] = 64
            r[1][1] = 64 - contrast_shift
            r[2][0] = 128 + 64
            r[2][1] = 128 + 64 + contrast_shift

            sorted(r, key=lambda v: v[0])
            hrzn, vrtc = list(zip(*r))
            return cls.adjustment_vector_estimator(hrzn, vrtc, adjustment_vector_size)

        def for_brightness_() -> np.ndarray:
            brightness_shift = np.abs(brightness) / 100
            rb = [[i / 3 * 255, i / 3 * 255] for i in range(4)]
            rb[1][0] = 130 - brightness_shift * 26
            rb[1][1] = 130 + brightness_shift * 51
            rb[2][0] = 233 - brightness_shift * 48
            rb[2][1] = 233 + brightness_shift * 10
            hrzn, vrtc = list(zip(*rb))
            vector = cls.adjustment_vector_estimator(hrzn, vrtc, adjustment_vector_size)

            # Adjust values for brightness lower then 0.
            if brightness < 0:
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

        return UInt8(adjustment_vector)


class PhotopeaLegacyReceiver(AbstractPhotopeaReceiver):
    @staticmethod
    def multiply(c, b):
        matrix = np.zeros(16)
        matrix[0] = c[0] * b[0] + c[1] * b[4] + c[2] * b[8] + c[3] * b[12]
        matrix[1] = c[0] * b[1] + c[1] * b[5] + c[2] * b[9] + c[3] * b[13]
        matrix[2] = c[0] * b[2] + c[1] * b[6] + c[2] * b[10] + c[3] * b[14]
        matrix[3] = c[0] * b[3] + c[1] * b[7] + c[2] * b[11] + c[3] * b[15]
        matrix[4] = c[4] * b[0] + c[5] * b[4] + c[6] * b[8] + c[7] * b[12]
        matrix[5] = c[4] * b[1] + c[5] * b[5] + c[6] * b[9] + c[7] * b[13]
        matrix[6] = c[4] * b[2] + c[5] * b[6] + c[6] * b[10] + c[7] * b[14]
        matrix[7] = c[4] * b[3] + c[5] * b[7] + c[6] * b[11] + c[7] * b[15]
        matrix[8] = c[8] * b[0] + c[9] * b[4] + c[10] * b[8] + c[11] * b[12]
        matrix[9] = c[8] * b[1] + c[9] * b[5] + c[10] * b[9] + c[11] * b[13]
        matrix[10] = c[8] * b[2] + c[9] * b[6] + c[10] * b[10] + c[11] * b[14]
        matrix[11] = c[8] * b[3] + c[9] * b[7] + c[10] * b[11] + c[11] * b[15]
        matrix[12] = c[12] * b[0] + c[13] * b[4] + c[14] * b[8] + c[15] * b[12]
        matrix[13] = c[12] * b[1] + c[13] * b[5] + c[14] * b[9] + c[15] * b[13]
        matrix[14] = c[12] * b[2] + c[13] * b[6] + c[14] * b[10] + c[15] * b[14]
        matrix[15] = c[12] * b[3] + c[13] * b[7] + c[14] * b[11] + c[15] * b[15]
        return matrix

    @staticmethod
    def clip(w):
        return np.clip(~~(w + 0.5).astype(Int), 0, 255)

    @classmethod
    def product_transform(cls, u, v, prodcut):
        v[0] = cls.clip(prodcut[0] * u[0] + prodcut[1] * u[1] + prodcut[2] * u[2] + prodcut[3] * 255)
        v[1] = cls.clip(prodcut[4] * u[0] + prodcut[5] * u[1] + prodcut[6] * u[2] + prodcut[7] * 255)
        v[2] = cls.clip(prodcut[8] * u[0] + prodcut[9] * u[1] + prodcut[10] * u[2] + prodcut[11] * 255)

    @classmethod
    def get_transformator(cls, brightness: Int, contrast: Int) -> Union[UInt8, np.ndarray]:
        brit = brightness / 255
        cntr = 1 + contrast / 100

        if cntr > 1:
            cntr = 1 + np.tan(np.pi / 2 * contrast / 101)

        h = (1 - cntr) / 2
        brit_transformer = [1, 0, 0, brit, 0, 1, 0, brit, 0, 0, 1, brit, 0, 0, 0, 1]
        cntr_transformer = [cntr, 0, 0, h, 0, cntr, 0, h, 0, 0, cntr, h, 0, 0, 0, 1]

        prodcut = cls.multiply(cntr_transformer, brit_transformer)
        transform_buffer = [np.arange(256, dtype=UInt8), np.zeros(256, dtype=UInt8), np.zeros(256, dtype=UInt8)]

        cls.product_transform(transform_buffer, transform_buffer, prodcut)

        return transform_buffer[0]


class StdReceiver(object):
    @classmethod
    def transform(cls, rgba: np.ndarray, brit: Float, cntr: Float, absolute: Bool = False) -> np.ndarray:
        """
        Mimic cv2.convertScaleAbs behavior when absolute is set as True.
        :param rgba:
        :param brit:
        :param cntr:
        :param absolute:
        :return:
        """
        adjusted = cntr * rgba.astype(float) + brit
        if absolute:
            adjusted = np.abs(adjusted)
        return np.clip(adjusted, 0, 255).astype(UInt8)


class AutoCntr(Command):
    def __init__(
            self,
            receiver: StdReceiver,
            rgba: np.ndarray,
            brit: Float,
            cntr: Float
    ) -> None:
        if not isinstance(rgba, np.ndarray) and rgba.dtype not in (UInt8,):
            raise TypeError('Invalid image type. Expected numpy.uint8 in shape h x w x 4')

        super(AutoCntr, self).__init__()
        self._rgba = rgba.copy()
        self._receiver = receiver
        self._brightness = brit
        self._contrast = cntr

    def execute(self) -> None:
        self._result = self._receiver.transform(rgba=self._rgba, brit=self._brightness, cntr=self._contrast)


class BritCntr(Command):
    def __init__(
            self,
            receiver: Union[PhotopeaReceiver, PhotopeaLegacyReceiver],
            rgba: np.ndarray,
            brit: Int,
            cntr: Int,
            legacy: Bool = False
    ) -> None:
        if not isinstance(rgba, np.ndarray) and rgba.dtype not in (UInt8,):
            raise TypeError('Invalid image type. Expected numpy.uint8 in shape h x w x 4')

        super(BritCntr, self).__init__()

        self._rgba = rgba.copy()
        self._receiver = receiver
        self._brightness = brit
        self._contrast = cntr
        self._legacy = legacy
        self._result = None

        self.validator()

    def validator(self):
        if not (-150 <= self._brightness <= 150):
            raise ValidationError('Invalid value for Brightness. Allowed interval is [-150, 150].')
        if not (-100 <= self._contrast <= 100):
            raise ValidationError('Invalid value for Contrast. Allowed inderval is [-100, 150]')

    def execute(self) -> None:
        receiver: PhotopeaReceiver = self._receiver
        if self._legacy:
            receiver: PhotopeaLegacyReceiver = self._receiver
        trasnformation_vector = receiver.transformation_vector(self._brightness, self._contrast)
        rgba = self._receiver.transform(self._rgba, tranformation_vector=trasnformation_vector)
        self._result = rgba


def brightness_contrast__(rgba: np.ndarray, brightness: int, contrast: int, use_legacy: bool = False) -> np.ndarray:
    """
    User facing method for Brightness/Contrast adjustment.
    Photoshop: Image -> Adjustment -> Brightness/Contrast

    :param rgba: np.ndarray;
    :param brightness: Int;
    :param contrast: Int;
    :param use_legacy: Bool;
    :return: np.ndarray;
    """
    receiver_cls = PhotopeaReceiver
    if use_legacy:
        receiver_cls = PhotopeaLegacyReceiver

    receiver = receiver_cls()
    command = BritCntr(rgba=rgba, receiver=receiver, brit=brightness, cntr=contrast, legacy=use_legacy)
    command.execute()
    return command.result


def brightness__(rgba: np.ndarray, brightness: int, use_legacy: bool = False) -> np.ndarray:
    """
    User facing method for Brightness adjustment.
    Mimic Photoshop: Edit -> Adjustment -> Brightness/Contrast

    :param rgba: np.ndarray;
    :param brightness: Int;
    :param use_legacy: Bool;
    :return: np.ndarray;
    """
    return brightness_contrast__(rgba=rgba, brightness=brightness, contrast=0, use_legacy=use_legacy)


def contrast__(rgba: np.ndarray, contrast: int, use_legacy: bool = False) -> np.ndarray:
    """
    User facing method for Contrast adjustment.
    Mimic Mimic Photoshop: Edit -> Adjustment -> Brightness/Contrast

    :param rgba: np.ndarray;
    :param contrast: Int;
    :param use_legacy: Bool;
    :return: np.ndarray;
    """
    return brightness_contrast__(rgba=rgba, brightness=0, contrast=contrast, use_legacy=use_legacy)


def auto_contrast__(rgba: np.ndarray):
    """
    Mimic Photoshop: Edit -> Adjustment -> Auto Contrast

    :param rgba: np.ndarray;
    :return: np.ndarray;
    """
    pass
