"""Gripper auto-calibration must hold the arm still while it drives the gripper to its end stops."""

import numpy as np

from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.tests.fake_motor_chain import FakeMotorChain
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml, detect_gripper_limits
from i2rt.utils.mujoco_utils import MuJoCoKDL

N = 7
GRIPPER = 6
TEST_TORQUE = 0.5
ARM_POS = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
STALE_EFF = np.array([0.0, 3.0, -2.0, 0.5, 0.0, 0.0, 0.0])  # whatever the motors happened to report at startup
KP = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0, 20.0])
KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5])


def _chain() -> FakeMotorChain:
    return FakeMotorChain(N, pos=np.append(ARM_POS, 0.02), eff=STALE_EFF, gripper_range=(0.0, 0.05))


def _calibration_frames(chain: FakeMotorChain) -> list:
    return [f for f in chain.frames if abs(f["torques"][GRIPPER]) == TEST_TORQUE]


def _calibrate(chain: FakeMotorChain, **hold: np.ndarray) -> tuple:
    return detect_gripper_limits(
        chain, GRIPPER, test_torque=TEST_TORQUE, max_duration=0.2, check_interval=0.001, **hold
    )


def test_calibration_holds_arm_with_pd_and_given_torques() -> None:
    chain = _chain()
    hold_torques = np.array([0.0, -0.7, 6.6, 1.6, 0.0, 0.0])

    limits = _calibrate(chain, arm_hold_kp=KP, arm_hold_kd=KD, arm_hold_torques=hold_torques)

    assert limits == (0.05, 0.0)
    frames = _calibration_frames(chain)
    assert {np.sign(f["torques"][GRIPPER]) for f in frames} == {1.0, -1.0}
    for f in frames:
        np.testing.assert_allclose(f["pos"][:GRIPPER], ARM_POS)
        np.testing.assert_allclose(f["kp"][:GRIPPER], KP[:GRIPPER])
        np.testing.assert_allclose(f["kd"][:GRIPPER], KD[:GRIPPER])
        np.testing.assert_allclose(f["torques"][:GRIPPER], hold_torques)
        assert f["kp"][GRIPPER] == 0.0 and f["kd"][GRIPPER] == 0.0


def test_calibration_stops_pushing_the_gripper_when_done() -> None:
    chain = _chain()

    _calibrate(chain, arm_hold_kp=KP, arm_hold_kd=KD)

    last = chain.last_frame()
    assert last["torques"][GRIPPER] == 0.0
    np.testing.assert_allclose(last["kp"][:GRIPPER], KP[:GRIPPER])


def test_calibration_without_hold_arguments_keeps_legacy_behaviour() -> None:
    chain = _chain()

    _calibrate(chain)

    for f in _calibration_frames(chain):
        np.testing.assert_allclose(f["kp"], 0.0)
        np.testing.assert_allclose(f["torques"][:GRIPPER], STALE_EFF[:GRIPPER])


def test_robot_calibration_holds_arm_against_model_gravity() -> None:
    xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    factor = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    kdl = MuJoCoKDL(xml)
    expected = kdl.compute_inverse_dynamics(ARM_POS, np.zeros(6), np.zeros(6)) * factor
    chain = _chain()

    robot = MotorChainRobot(
        chain,
        xml_path=xml,
        use_gravity_comp=True,
        gravity_comp_factor=np.append(factor, 1.0),
        joint_limits=kdl.joint_limits[:GRIPPER],  # arm joints only, as get_yam_robot passes them
        gripper_index=GRIPPER,
        kp=KP.tolist(),
        kd=KD.tolist(),
        enable_gripper_calibration=True,
        test_torque=TEST_TORQUE,
        test_duration=0.2,
        check_interval=0.001,
    )
    try:
        frames = _calibration_frames(chain)
        assert frames
        for f in frames:
            np.testing.assert_allclose(f["torques"][:GRIPPER], expected, atol=1e-9)
            np.testing.assert_allclose(f["kp"][:GRIPPER], KP[:GRIPPER])
            np.testing.assert_allclose(f["pos"][:GRIPPER], ARM_POS)
        np.testing.assert_allclose(sorted(robot.get_robot_info()["gripper_limits"]), [0.0, 0.05])
    finally:
        robot.close()
