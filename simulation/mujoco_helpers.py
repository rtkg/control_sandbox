import mujoco
import numpy as np


def mjc_qpos_idx(self, joint_names: list) -> np.ndarray:
    # get the MuJoCo data.qpos indices of the given joints
    mjc_qpos_idx = np.zeros(len(joint_names), dtype=int)
    for i, name in enumerate(joint_names):
        idx = self._model.joint(name).qposadr
        # Fail if the corresponding joint is not one-dimensional (hinge/slider)
        if len(idx) != 1:
            raise RuntimeError(f"[MuJoCo]: Joint {name} is invalid.")

        mjc_qpos_idx[i] = idx[0]

    return mjc_qpos_idx


def mjc_qvel_idx(self, joint_names: list) -> np.ndarray:
    # get the MuJoCo data.qvel indices of the given joints

    # compute the index list mapping from the given joints top MuJoCo joint velocity indices
    mjc_qvel_idx = np.zeros(len(joint_names), dtype=int)
    for i, name in enumerate(joint_names):
        idx = self._model.joint(name).dofadr
        # Fail if the corresponding joint is not one-dimensional (hinge/slider)
        if len(idx) != 1:
            raise RuntimeError(f"[MuJoCo]: Joint {name} is invalid.")

        mjc_qvel_idx[i] = idx[0]

    return mjc_qvel_idx


def mjc_ctrl_idx(self, joint_names: list) -> np.ndarray:
    # get the MuJoCo data.ctrl indices of the actuators driving the given joints
    n_joints = len(joint_names)
    mjc_ctrl_idx = np.zeros(n_joints, dtype=int)
    for i, name in enumerate(joint_names):
        # find the actuator acting on the current joint
        actuator_found = False
        for j in range(n_joints):
            act_name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, j
            )
            jnt_id = self._model.actuator(act_name).trnid[0]
            jnt_name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id
            )
            if jnt_name == name:
                mjc_ctrl_idx[i] = j
                actuator_found = True
                break

        if not actuator_found:
            raise RuntimeError(
                f"[MuJoCo]: No actuator found for joint {name}."
            )

    return mjc_ctrl_idx
