from __future__ import annotations

import unittest

import numpy as np

from rose.protocol import make_one_view_drop_mask


class MaskProtocolTests(unittest.TestCase):
    def test_exact_incomplete_count_and_one_drop(self) -> None:
        mask = make_one_view_drop_mask(101, 4, 0.30, 3)
        incomplete = np.any(mask == 0, axis=1)
        self.assertEqual(int(incomplete.sum()), 30)
        self.assertTrue(np.all(np.sum(mask[incomplete] == 0, axis=1) == 1))

    def test_deterministic(self) -> None:
        left = make_one_view_drop_mask(120, 3, 0.50, 2)
        right = make_one_view_drop_mask(120, 3, 0.50, 2)
        np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
