"""Record measured joint angles and export six curves without a GUI."""

import csv
from datetime import datetime
import math
from pathlib import Path
import time
from collections.abc import Sequence

import numpy as np


class JointAngleRecorder:
    """Stream feedback to CSV and plot angles in radians on a fixed +/-2pi axis."""

    def __init__(
        self,
        output_dir: Path,
        joint_names: Sequence[str],
        sample_period_s: float,
        mode: str,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self.sample_period_s = sample_period_s
        self._start_time: float | None = None
        self._last_sample_time: float | None = None
        self._last_flush_time: float | None = None
        self._samples: list[tuple[float, ...]] = []
        self._plotted_sample_count = 0

        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.csv_path = output_dir / f"joint_angles_{stamp}_{mode}.csv"
        self.png_path = self.csv_path.with_suffix(".png")
        self._file = self.csv_path.open("x", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time_s", *(f"{name}_rad" for name in self.joint_names)])
        self._file.flush()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def record(self, joint_names: Sequence[str], positions: Sequence[float]) -> bool:
        """Record complete, finite feedback at most once per sampling period."""
        if self._file.closed:
            return False
        now = time.monotonic()
        if self._last_sample_time is not None and now - self._last_sample_time < self.sample_period_s:
            return False

        indices = {name: index for index, name in enumerate(joint_names)}
        try:
            angles = np.asarray([positions[indices[name]] for name in self.joint_names], dtype=np.float64)
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        if not np.all(np.isfinite(angles)):
            return False

        if self._start_time is None:
            self._start_time = now
        # 保留真实反馈，不做限幅或角度取模，以免掩盖异常或制造曲线跳变。
        sample = (now - self._start_time, *angles.tolist())
        self._writer.writerow(sample)
        self._samples.append(sample)
        self._last_sample_time = now
        if self._last_flush_time is None or now - self._last_flush_time >= 1.0:
            self._file.flush()
            self._last_flush_time = now
        return True

    def save_plot(self) -> Path | None:
        """Save all recorded feedback, including samples after an episode reset."""
        if not self._file.closed:
            self._file.flush()
        if not self._samples:
            return None
        if self._plotted_sample_count == self.sample_count:
            return self.png_path

        # 使用 Agg 画布直接写文件，不创建窗口或改变其他绘图功能的后端。
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        data = np.asarray(self._samples, dtype=np.float64)
        figure = Figure(figsize=(12.0, 6.0))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        ticks = [-2.0 * math.pi, -math.pi, 0.0, math.pi, 2.0 * math.pi]
        tick_labels = [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"]
        try:
            for index, name in enumerate(self.joint_names):
                axis.plot(
                    data[:, 0],
                    data[:, index + 1],
                    color=f"C{index}",
                    linewidth=1.5,
                    marker="." if self.sample_count == 1 else None,
                    label=f"J{index + 1}: {name}",
                )
            axis.set_title("Measured Joint Angles")
            axis.set_xlabel("Elapsed time (s)")
            axis.set_ylabel("Joint angle (rad)")
            axis.set_ylim(-2.0 * math.pi, 2.0 * math.pi)
            axis.set_yticks(ticks, tick_labels)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right", ncol=2)
            figure.tight_layout()
            figure.savefig(self.png_path, dpi=180)
        finally:
            figure.clear()
        self._plotted_sample_count = self.sample_count
        return self.png_path

    def close(self) -> Path | None:
        """Close the CSV before plotting so data survives a plotting failure."""
        self._file.close()
        return self.save_plot()
