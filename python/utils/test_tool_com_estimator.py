import unittest

import numpy as np

from python.utils.tool_com_estimator import fit_tool_com, transform_point


class ToolComEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        directions = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        self.gravity = 9.80665 * directions
        self.mass = 0.73
        self.com = np.array([0.012, -0.028, 0.164], dtype=np.float64)
        self.force_bias = np.array([0.2, -0.1, 0.3], dtype=np.float64)
        self.torque_bias = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    def make_wrenches(self, sign: float) -> np.ndarray:
        load_force = sign * self.mass * self.gravity
        force = self.force_bias + load_force
        torque = np.stack(
            [self.torque_bias + np.cross(self.com, force_i) for force_i in load_force]
        )
        return np.column_stack([force, torque])

    def test_recovers_mass_com_and_bias(self) -> None:
        fit = fit_tool_com(self.gravity, self.make_wrenches(sign=1.0))

        self.assertAlmostEqual(fit.mass_kg, self.mass, places=12)
        np.testing.assert_allclose(fit.com_offset_m, self.com, atol=1.0e-12)
        np.testing.assert_allclose(fit.force_bias_n, self.force_bias, atol=1.0e-12)
        np.testing.assert_allclose(fit.torque_bias_nm, self.torque_bias, atol=1.0e-12)

    def test_reaction_wrench_sign_does_not_flip_com(self) -> None:
        fit = fit_tool_com(self.gravity, self.make_wrenches(sign=-1.0))

        self.assertAlmostEqual(fit.mass_kg, self.mass, places=12)
        self.assertLess(fit.signed_mass_kg, 0.0)
        np.testing.assert_allclose(fit.com_offset_m, self.com, atol=1.0e-12)

    def test_small_noise_stays_close_to_ground_truth(self) -> None:
        rng = np.random.default_rng(7)
        wrench = self.make_wrenches(sign=1.0)
        wrench[:, :3] += rng.normal(0.0, 0.015, size=(wrench.shape[0], 3))
        wrench[:, 3:] += rng.normal(0.0, 0.001, size=(wrench.shape[0], 3))

        fit = fit_tool_com(self.gravity, wrench)

        self.assertAlmostEqual(fit.mass_kg, self.mass, delta=0.003)
        np.testing.assert_allclose(fit.com_offset_m, self.com, atol=5.0e-4)

    def test_rejects_repeated_gravity_direction(self) -> None:
        gravity = np.repeat([[0.0, 0.0, -9.80665]], repeats=6, axis=0)
        wrench = np.zeros((6, 6), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "Gravity directions"):
            fit_tool_com(gravity, wrench)

    def test_transforms_com_point_to_sim_frame(self) -> None:
        half_angle = np.pi / 4.0
        quaternion_z_90_xyzw = np.array(
            [0.0, 0.0, np.sin(half_angle), np.cos(half_angle)]
        )
        transformed = transform_point(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            quaternion_z_90_xyzw,
        )

        np.testing.assert_allclose(transformed, [0.1, 1.2, 0.3], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()

