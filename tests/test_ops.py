from photoshop.ops.transform import shift_scale

from tests.utils import DataTestCase


class TransformTestCase(DataTestCase):
    # noinspection PyMethodMayBeStatic
    def test_shift_scale(self):
        expected_value = [35, 35.75, 40]
        shifted_middle = shift_scale(0.0, old_scale=(-100, 100), new_scale=(30, 40))
        shifted_random = shift_scale(15, old_scale=(-100, 100), new_scale=(30, 40))
        shifted_boundr = shift_scale(100, old_scale=(-100, 100), new_scale=(30, 40))

        assert shifted_middle, expected_value[0]
        assert shifted_random, expected_value[1]
        assert shifted_boundr, expected_value[2]

    def test_uint32_to_rgba(self):
        pass

    def rgba_to_gray(self):
        pass
