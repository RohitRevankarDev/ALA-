"""
test_vector.py
Unit tests for Vector.mean, Vector.demean and Vector.std.

Run with either:
    python -m unittest test_vector -v
    pytest -v
"""

import math
import random
import unittest

from vector import Vector


class TestMean(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(Vector([1, 2, 3, 4, 5]).mean(), 3.0)
        self.assertAlmostEqual(Vector([2, 4, 6, 8]).mean(), 5.0)
        self.assertAlmostEqual(Vector([-1, 1]).mean(), 0.0)

    def test_single_entry(self):
        self.assertAlmostEqual(Vector([7.5]).mean(), 7.5)

    def test_constant_vector(self):
        self.assertAlmostEqual(Vector([4, 4, 4, 4]).mean(), 4.0)

    def test_negative_and_float_entries(self):
        self.assertAlmostEqual(Vector([-2.5, 0.5, 1.0]).mean(), -1.0 / 3)

    def test_equals_ones_dot_x_over_n(self):
        x = Vector([3, -1, 4, 1, 5])
        ones = Vector([1] * len(x))
        self.assertAlmostEqual(x.mean(), ones.dot(x) / len(x))

    def test_linearity_scaling(self):
        x = Vector([1, 2, 3, 10])
        a = -2.5
        self.assertAlmostEqual((a * x).mean(), a * x.mean())

    def test_linearity_addition(self):
        x = Vector([1, 2, 3, 10])
        y = Vector([-4, 0, 7, 2])
        self.assertAlmostEqual((x + y).mean(), x.mean() + y.mean())

    def test_shift_property(self):
        x = Vector([1, 2, 3, 10])
        c = 3.7
        shifted = Vector(v + c for v in x)
        self.assertAlmostEqual(shifted.mean(), x.mean() + c)

    def test_mean_lies_between_min_and_max(self):
        x = Vector([-3, 8, 2, 0.5, 11])
        self.assertLessEqual(min(x), x.mean())
        self.assertLessEqual(x.mean(), max(x))

    def test_empty_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector([]).mean()


class TestDemean(unittest.TestCase):
    def test_known_values(self):
        d = Vector([1, 2, 3]).demean()
        for got, want in zip(d, [-1.0, 0.0, 1.0]):
            self.assertAlmostEqual(got, want)

    def test_returns_new_vector_and_does_not_modify_original(self):
        x = Vector([5, 6, 7])
        d = x.demean()
        self.assertIsInstance(d, Vector)
        self.assertIsNot(d, x)
        self.assertEqual(x.entries, [5.0, 6.0, 7.0])

    def test_same_length(self):
        x = Vector([1, 9, -4, 2])
        self.assertEqual(len(x.demean()), len(x))

    def test_mean_of_demeaned_is_zero(self):
        x = Vector([3, -1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(x.demean().mean(), 0.0)

    def test_demean_of_constant_vector_is_zero_vector(self):
        d = Vector([6, 6, 6, 6]).demean()
        for v in d:
            self.assertAlmostEqual(v, 0.0)

    def test_idempotent(self):
        x = Vector([3, -1, 4, 1, 5])
        once = x.demean()
        twice = once.demean()
        for a, b in zip(once, twice):
            self.assertAlmostEqual(a, b)

    def test_x_equals_demeaned_plus_mean(self):
        x = Vector([3, -1, 4, 1, 5])
        m = x.mean()
        rebuilt = Vector(d + m for d in x.demean())
        for a, b in zip(x, rebuilt):
            self.assertAlmostEqual(a, b)

    def test_demean_is_shift_invariant(self):
        x = Vector([1, 2, 3, 10])
        shifted = Vector(v + 100 for v in x)
        for a, b in zip(x.demean(), shifted.demean()):
            self.assertAlmostEqual(a, b)

    def test_demean_scales_linearly(self):
        x = Vector([1, 2, 3, 10])
        a = 3.0
        for u, v in zip((a * x).demean(), a * x.demean()):
            self.assertAlmostEqual(u, v)

    def test_demeaned_is_orthogonal_to_ones(self):
        x = Vector([3, -1, 4, 1, 5])
        ones = Vector([1] * len(x))
        self.assertAlmostEqual(x.demean().dot(ones), 0.0)

    def test_empty_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector([]).demean()


class TestStd(unittest.TestCase):
    def test_known_values(self):
        # deviations -1, 0, 1 -> sqrt(2/3)
        self.assertAlmostEqual(Vector([1, 2, 3]).std(), math.sqrt(2 / 3))
        # classic example: population std = 2
        self.assertAlmostEqual(Vector([2, 4, 4, 4, 5, 5, 7, 9]).std(), 2.0)

    def test_single_entry_is_zero(self):
        self.assertAlmostEqual(Vector([42]).std(), 0.0)

    def test_constant_vector_has_zero_std(self):
        self.assertAlmostEqual(Vector([3, 3, 3, 3]).std(), 0.0)

    def test_std_zero_only_if_constant(self):
        self.assertGreater(Vector([1, 1, 1, 2]).std(), 0.0)

    def test_non_negative(self):
        for data in ([1, 2, 3], [-5, 5], [0.1, 0.2, 0.9], [100, -100, 0]):
            self.assertGreaterEqual(Vector(data).std(), 0.0)

    def test_equals_rms_of_demeaned_vector(self):
        x = Vector([3, -1, 4, 1, 5])
        self.assertAlmostEqual(x.std(), x.demean().rms())

    def test_equals_norm_of_demeaned_over_sqrt_n(self):
        x = Vector([3, -1, 4, 1, 5])
        self.assertAlmostEqual(x.std(), x.demean().norm() / math.sqrt(len(x)))

    def test_shift_invariance(self):
        x = Vector([1, 2, 3, 10])
        shifted = Vector(v - 55.5 for v in x)
        self.assertAlmostEqual(x.std(), shifted.std())

    def test_scaling_property(self):
        x = Vector([1, 2, 3, 10])
        for a in (2.0, -2.0, 0.5, -0.25):
            self.assertAlmostEqual((a * x).std(), abs(a) * x.std())

    def test_rms_identity(self):
        # rms(x)^2 = mean(x)^2 + std(x)^2
        for data in ([1, 2, 3, 10], [-4, 7, 0, 2, 9], [5, 5, 5]):
            x = Vector(data)
            self.assertAlmostEqual(x.rms() ** 2, x.mean() ** 2 + x.std() ** 2)

    def test_std_at_most_rms(self):
        x = Vector([3, -1, 4, 1, 5])
        self.assertLessEqual(x.std(), x.rms() + 1e-12)

    def test_chebyshev_inequality(self):
        # fraction of entries with |x_i - mean| >= k*std is at most 1/k^2
        random.seed(0)
        x = Vector([random.gauss(10, 4) for _ in range(500)])
        m, s = x.mean(), x.std()
        for k in (1.5, 2, 3):
            frac = sum(1 for v in x if abs(v - m) >= k * s) / len(x)
            self.assertLessEqual(frac, 1 / k**2 + 1e-12)

    def test_std_is_standardisation_denominator(self):
        # standardised vector has mean 0 and std 1
        x = Vector([3, -1, 4, 1, 5, 9])
        z = x.demean() * (1 / x.std())
        self.assertAlmostEqual(z.mean(), 0.0)
        self.assertAlmostEqual(z.std(), 1.0)

    def test_empty_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector([]).std()


if __name__ == "__main__":
    unittest.main()
