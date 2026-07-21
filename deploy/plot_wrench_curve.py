#!/usr/bin/env python3
"""Live plot the UR wrist wrench topic while deployment is running."""

from __future__ import annotations

import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node


WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
HISTORY_SECONDS = 30.0
PLOT_HZ = 20.0


class WrenchCurveNode(Node):
    def __init__(self) -> None:
        super().__init__("wrench_curve_plot_node")
        self.start_time = time.monotonic()
        self.times: deque[float] = deque()
        self.wrenches: deque[np.ndarray] = deque()
        self.create_subscription(WrenchStamped, WRENCH_TOPIC, self.wrench_callback, 50)
        self.get_logger().info(f"Plotting live wrench curve from {WRENCH_TOPIC}")

    def wrench_callback(self, msg: WrenchStamped) -> None:
        now = time.monotonic() - self.start_time
        wrench = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=np.float32,
        )
        self.times.append(now)
        self.wrenches.append(wrench)
        while self.times and now - self.times[0] > HISTORY_SECONDS:
            self.times.popleft()
            self.wrenches.popleft()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.times:
            return np.empty(0, dtype=np.float32), np.empty((0, 6), dtype=np.float32)
        return np.asarray(self.times, dtype=np.float32), np.asarray(self.wrenches, dtype=np.float32)


def update_axis(ax, lines, x: np.ndarray, values: np.ndarray, ylabel: str) -> None:
    for line, y in zip(lines, values.T):
        line.set_data(x, y)
    ax.set_xlim(min(-HISTORY_SECONDS, float(x[0])), 0.0)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    ax.set_ylabel(ylabel)


def main() -> None:
    rclpy.init()
    node = WrenchCurveNode()

    plt.ion()
    fig, (ax_force, ax_torque) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.canvas.manager.set_window_title("UR wrist wrench curve")

    force_labels = ["Fx", "Fy", "Fz", "|F|"]
    torque_labels = ["Tx", "Ty", "Tz", "|T|"]
    force_lines = [ax_force.plot([], [], label=label)[0] for label in force_labels]
    torque_lines = [ax_torque.plot([], [], label=label)[0] for label in torque_labels]

    ax_force.grid(True, alpha=0.3)
    ax_torque.grid(True, alpha=0.3)
    ax_force.legend(loc="upper left")
    ax_torque.legend(loc="upper left")
    ax_torque.set_xlabel(f"time from now (s), last {HISTORY_SECONDS:.0f}s")

    last_plot_time = 0.0
    try:
        while rclpy.ok() and plt.fignum_exists(fig.number):
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now - last_plot_time < 1.0 / PLOT_HZ:
                continue
            last_plot_time = now

            times, wrenches = node.snapshot()
            if times.size == 0:
                plt.pause(0.001)
                continue

            x = times - times[-1]
            force = wrenches[:, 0:3]
            torque = wrenches[:, 3:6]
            force_plot = np.column_stack([force, np.linalg.norm(force, axis=1)])
            torque_plot = np.column_stack([torque, np.linalg.norm(torque, axis=1)])

            update_axis(ax_force, force_lines, x, force_plot, "force (N)")
            update_axis(ax_torque, torque_lines, x, torque_plot, "torque (Nm)")
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
