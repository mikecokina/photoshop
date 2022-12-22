from typing import Union

from photoshop.libs.numpy import np
from photoshop.core.dtype import Int, UInt8, UInt32


def uint32_to_rgba(uint32_img: Union[UInt32, np.ndarray], width: Int, height: Int) -> Union[UInt8, np.ndarray]:
    r = UInt8(uint32_img)
    g = UInt8(uint32_img >> 8)
    b = UInt8(uint32_img >> 16)
    a = UInt8(uint32_img >> 24)
    del uint32_img
    return np.stack([r, g, b, a]).T.reshape(height, width, 4)
