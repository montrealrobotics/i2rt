"""MotorChainRobot command/state API, exercised against an in-memory motor chain (no CAN bus)."""

import time
from typing import Iterator

import numpy as np
import pytest

from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.tests.fake_motor_chain import FakeMotorChain

N = 6
KP = [80.0, 80.0, 80.0, 10.0, 10.0, 10.0]
KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
LIMITS = np.array([[-3.0, 3.0]] * N)


def _wait_for_frame(chain: FakeMotorChain, after: int) -> dict:
    """Return the first command frame sent after ``after`` frames had been recorded."""
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if len(chain.frames) > after + 1:
            return chain.last_frame()
        time.sleep(0.002)
    raise TimeoutError("control thread sent no new frame")


@pytest.fixture
def chain() -> FakeMotorChain:
    return FakeMotorChain(N, pos=np.linspace(-0.5, 0.5, N), eff=np.arange(N, dtype=float))


@pytest.fixture
def robot(chain: FakeMotorChain) -> Iterator[MotorChainRobot]:
    r = MotorChainRobot(chain, use_gravity_comp=False, joint_limits=LIMITS, kp=KP, kd=KD)
    yield r
    r.close()


def test_get_joint_state_returns_pos_vel_eff(robot: MotorChainRobot, chain: FakeMotorChain) -> None:
    state = robot.get_joint_state()

    assert state is not None
    np.testing.assert_allclose(state["pos"], chain.pos)
    np.testing.assert_allclose(state["vel"], chain.vel)
    np.testing.assert_allclose(state["eff"], chain.eff)
