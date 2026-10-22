"""MotorChainRobot command/state API, exercised against an in-memory motor chain (no CAN bus)."""

import threading
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


def test_command_target_vel_sends_damping_only_velocity_command(robot: MotorChainRobot, chain: FakeMotorChain) -> None:
    target = np.array([0.5, -0.5, 0.2, -0.2, 0.1, -0.1])

    robot.command_target_vel(target)
    frame = _wait_for_frame(chain, len(chain.frames))

    np.testing.assert_allclose(frame["vel"], target)
    np.testing.assert_allclose(frame["kp"], 0.0)
    np.testing.assert_allclose(frame["kd"], KD)


def test_command_target_vel_does_not_drive_a_joint_past_its_limit(
    robot: MotorChainRobot, chain: FakeMotorChain
) -> None:
    chain.pos[0] = LIMITS[0, 1]  # joint 1 at its upper limit
    chain.pos[1] = LIMITS[1, 0]  # joint 2 at its lower limit
    _wait_for_frame(chain, len(chain.frames))

    robot.command_target_vel(np.array([0.5, -0.5, 0.2, 0.0, 0.0, 0.0]))
    frame = _wait_for_frame(chain, len(chain.frames))

    np.testing.assert_allclose(frame["vel"], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0])


def test_command_target_vel_still_allows_moving_back_into_range(robot: MotorChainRobot, chain: FakeMotorChain) -> None:
    chain.pos[0] = LIMITS[0, 1]
    _wait_for_frame(chain, len(chain.frames))

    robot.command_target_vel(np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    frame = _wait_for_frame(chain, len(chain.frames))

    assert frame["vel"][0] == pytest.approx(-0.5)


def test_command_joint_state_never_publishes_a_half_built_command(robot: MotorChainRobot) -> None:
    """While the control thread holds the command lock, command_joint_state must not touch self._commands."""
    robot.command_joint_pos(np.zeros(N))
    before = robot._commands
    target = {"pos": np.full(N, 0.3), "vel": np.zeros(N)}

    with robot._command_lock:
        writer = threading.Thread(target=robot.command_joint_state, args=(target,))
        writer.start()
        writer.join(timeout=0.05)
        assert robot._commands is before
        np.testing.assert_allclose(robot._commands.kp, KP)

    writer.join(timeout=1.0)
    np.testing.assert_allclose(robot._commands.pos, 0.3)
    np.testing.assert_allclose(robot._commands.kp, KP)
    np.testing.assert_allclose(robot._commands.kd, KD)
