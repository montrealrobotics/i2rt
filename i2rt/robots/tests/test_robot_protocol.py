"""The Robot protocol's optional methods must fail loudly when a robot does not implement them."""

from typing import Dict

import numpy as np
import pytest

from i2rt.robots.robot import Robot


class _MinimalRobot(Robot):
    def num_dofs(self) -> int:
        return 2

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {}


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("get_joint_pos", ()),
        ("get_joint_state", ()),
        ("command_joint_pos", (np.zeros(2),)),
        ("command_target_vel", (np.zeros(2),)),
        ("command_joint_state", ({"pos": np.zeros(2), "vel": np.zeros(2)},)),
    ],
)
def test_unimplemented_methods_raise(method: str, args: tuple) -> None:
    with pytest.raises(NotImplementedError):
        getattr(_MinimalRobot(), method)(*args)


def test_lifecycle_hooks_stay_optional_no_ops() -> None:
    robot = _MinimalRobot()
    robot.reinit()
    robot.close()
