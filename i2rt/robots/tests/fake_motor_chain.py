"""In-memory stand-in for a CAN motor chain, so MotorChainRobot can be built without hardware."""

import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from i2rt.motor_drivers.dm_driver import MotorInfo


class FakeMotorChain:
    """Records every command frame and reports a fixed (settable) motor state."""

    def __init__(
        self,
        n: int,
        pos: Optional[np.ndarray] = None,
        eff: Optional[np.ndarray] = None,
        gripper_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """``gripper_range``: if set, the last motor is a gripper that jumps to the end of this range
        in the direction of any torque commanded on it, as when calibration drives it to its end stops."""
        self.motor_list = [[i + 1, "DM4310"] for i in range(n)]
        self.motor_direction = np.ones(n)
        self.running = True
        self.enable_auto_recovery = False
        self.same_bus_device_driver = None
        self.pos = np.zeros(n) if pos is None else np.asarray(pos, dtype=float)
        self.vel = np.zeros(n)
        self.eff = np.zeros(n) if eff is None else np.asarray(eff, dtype=float)
        self.gripper_range = gripper_range
        self.frames: List[dict] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.motor_list)

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        return [
            MotorInfo(id=i + 1, error_code=1, pos=self.pos[i], vel=self.vel[i], eff=self.eff[i], timestamp=time.time())
            for i in range(len(self))
        ]

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> List[MotorInfo]:
        n = len(self)
        frame = {
            "torques": np.array(torques, dtype=float),
            "pos": np.zeros(n) if pos is None else np.array(pos, dtype=float),
            "vel": np.zeros(n) if vel is None else np.array(vel, dtype=float),
            "kp": np.zeros(n) if kp is None else np.array(kp, dtype=float),
            "kd": np.zeros(n) if kd is None else np.array(kd, dtype=float),
        }
        with self._lock:
            self.frames.append(frame)
        if self.gripper_range is not None and frame["torques"][-1] != 0.0:
            self.pos[-1] = max(self.gripper_range) if frame["torques"][-1] > 0 else min(self.gripper_range)
        return self.read_states()

    def last_frame(self) -> dict:
        with self._lock:
            return self.frames[-1]

    def close(self) -> None:
        self.running = False
