import numpy as np

from ..core.dtype import Int


def histogram_clip(ndarray: np.ndarray, clip_hist_percent: Int):
    """
    Determine minimal and maximal value to be cliped out in image of 0-255 colors, based on color histogram tails.

        minimum - left histogram tail (all colors bellow)
        maximum - right histogam tail (all colors above)

    :param ndarray: np.ndarray;
    :param clip_hist_percent: Numeric;
    :return: Tuple[Int, Int];
    """
    # Calculate histogram of gray scaled image.
    hist, _ = np.histogram(ndarray, bins=256)
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = np.cumsum(hist)

    # Locate points to clip.
    # Idea is to remove too white/black pixels (set them 0/255) if their currendistribution is bellow
    # given threshold (clip_hist_percent).
    clip_hist_percent *= (accumulator.max() / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum = 0
    while accumulator[minimum] < clip_hist_percent:
        minimum += 1

    # Locate right cut
    maximum = hist_size - 1
    while accumulator[maximum] >= (accumulator.max() - clip_hist_percent):
        maximum -= 1
    return minimum, maximum
