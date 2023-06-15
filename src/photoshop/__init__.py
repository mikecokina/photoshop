from .adjustments.brightness_contrast import brightness__
from .adjustments.brightness_contrast import contrast__
from .adjustments.brightness_contrast import brightness_contrast__
from .adjustments.brightness_contrast import auto_contrast__

from .filters.distort.displacement import displacement__


class Adjustments(object):
    brigthness = brightness__
    contrast = contrast__
    brightness_contrast = brightness_contrast__
    auto_contrast = auto_contrast__


class Filters(object):
    class Distort(object):
        displacement = displacement__

    distort = Distort()


adjustments = Adjustments()
filters = Filters()


__version__ = '0.0.0.dev0'


__all__ = (
    'adjustments',
    'filters'
)
