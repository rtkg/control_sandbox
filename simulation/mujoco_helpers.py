import mujoco
import numpy as np
import collections
from copy import deepcopy, copy
import logging


def mjc_body_jacobian(model, data):
    """
    Compute the body jacobian which relates joint velocities to the body twist (i.e.,
    the end-effector linear, angular velocity expressed in the end-effector frame).
    Currently, this method is hardcoded for the model of the Franka Emika Panda robot.
    It is assumed, that the kinematic quantities in 'data' are up-to date, i.e., that
    at least mj_forward(.) was called prior to this function.

    Args:
        model ... MuJoCo model struct
        data ... Mujoco data struct

    returns
        jac ... body jacobian
    """
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    # compute position/rotation parts of the jacobian
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site("panda_tool_center_point").id)

    # MuJoCo returns the body jacobian expressed in the world frame, this needs to be
    # rotated to the end-effector frame
    R_ew = data.site("panda_tool_center_point").xmat.reshape(3, 3).transpose()

    return np.vstack((R_ew @ jacp[:, 0:7], R_ew @ jacr[:, 0:7]))


def mjc_world_jacobian(model, data):
    """
    Compute the body jacobian which relates joint velocities to the body twist (i.e.,
    the end-effector linear, angular velocity expressed in the world frame).
    Currently, this method is hardcoded for the model of the Franka Emika Panda robot.
    It is assumed, that the kinematic quantities in 'data' are up-to date, i.e., that
    at least mj_forward(.) was called prior to this function.

    Args:
        model ... MuJoCo model struct
        data ... Mujoco data struct

    returns
        jac ... body jacobian
    """
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    # compute position/rotation parts of the jacobian
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site("panda_tool_center_point").id)

    return np.vstack((jacp[:, 0:7], jacr[:, 0:7]))


def mjc_body_jacobian_derivative(model, data):
    """
    Computes the body jacobian derivative via forward simulation by a small timestep
    and numerical differentiation as suggested here: https://github.com/google-deepmind/mujoco/issues/411.
    Currently, this method is hardcoded for the model of the Franka Emika Panda robot.
    It is assumed, that the kinematic quantities in 'data' are up-to date, i.e., that
    at least mj_forward(.) was called prior to this function.

    Args:
        model ... MuJoCo model struct
        data ... Mujoco data struct

    returns
        jac_dot ... time derivative of the body jacobian
    """

    h = 1e-10
    J = mjc_body_jacobian(model, data)

    datah = copy(data)

    mujoco.mj_integratePos(model, datah.qpos, datah.qvel, h)
    mujoco.mj_fwdPosition(model, datah)
    Jh = mjc_body_jacobian(model, datah)
    Jdot = (Jh - J) / h

    return Jdot


def mjc_world_jacobian_derivative(model, data):
    """
    Computes the body jacobian derivative (expressed in the world frame) via forward simulation by a small timestep
    and numerical differentiation as suggested here: https://github.com/google-deepmind/mujoco/issues/411.
    Currently, this method is hardcoded for the model of the Franka Emika Panda robot.
    It is assumed, that the kinematic quantities in 'data' are up-to date, i.e., that
    at least mj_forward(.) was called prior to this function.

    Args:
        model ... MuJoCo model struct
        data ... Mujoco data struct

    returns
        jac_dot ... time derivative of the body jacobian
    """

    h = 1e-10
    J = mjc_world_jacobian(model, data)

    datah = copy(data)
    mujoco.mj_integratePos(model, datah.qpos, datah.qvel, h)
    mujoco.mj_fwdPosition(model, datah)
    Jh = mjc_world_jacobian(model, datah)
    Jdot = (Jh - J) / h

    return Jdot


def mjc_qpos_idx(model, joint_names: list) -> np.ndarray:
    # get the MuJoCo data.qpos indices of the given joints
    mjc_qpos_idx = np.zeros(len(joint_names), dtype=int)
    for i, name in enumerate(joint_names):
        idx = model.joint(name).qposadr
        # Fail if the corresponding joint is not one-dimensional (hinge/slider)
        if len(idx) != 1:
            raise RuntimeError(f"[MuJoCo]: Joint {name} is invalid.")

        mjc_qpos_idx[i] = idx[0]

    return mjc_qpos_idx


def mjc_qvel_idx(model, joint_names: list) -> np.ndarray:
    # get the MuJoCo data.qvel indices of the given joints

    # compute the index list mapping from the given joints top MuJoCo joint velocity indices
    mjc_qvel_idx = np.zeros(len(joint_names), dtype=int)
    for i, name in enumerate(joint_names):
        idx = model.joint(name).dofadr
        # Fail if the corresponding joint is not one-dimensional (hinge/slider)
        if len(idx) != 1:
            raise RuntimeError(f"[MuJoCo]: Joint {name} is invalid.")

        mjc_qvel_idx[i] = idx[0]

    return mjc_qvel_idx


def mjc_ctrl_idx(model, joint_names: list) -> np.ndarray:
    # get the MuJoCo data.ctrl indices of the actuators driving the given joints
    n_joints = len(joint_names)
    mjc_ctrl_idx = np.zeros(n_joints, dtype=int)
    for i, name in enumerate(joint_names):
        # find the actuator acting on the current joint
        actuator_found = False
        for j in range(n_joints):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
            jnt_id = model.actuator(act_name).trnid[0]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name == name:
                mjc_ctrl_idx[i] = j
                actuator_found = True
                break

        if not actuator_found:
            raise RuntimeError(f"[MuJoCo]: No actuator found for joint {name}.")

    return mjc_ctrl_idx


# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Functions for computing inverse kinematics on MuJoCo models.
Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py"""

_INVALID_JOINT_NAMES_TYPE = (
    "`joint_names` must be either None, a list, a tuple, or a numpy array; " "got {}."
)
_REQUIRE_TARGET_POS_OR_QUAT = (
    "At least one of `target_pos` or `target_quat` must be specified."
)

IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])


def qpos_from_site_pose(
    model,
    data,
    site_name,
    target_pos=None,
    target_quat=None,
    joint_names=None,
    tol=1e-14,
    rot_weight=1.0,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
    max_update_norm=2.0,
    progress_thresh=20.0,
    max_steps=100,
    inplace=False,
):
    """Find joint positions that satisfy a target site position and/or rotation.

    Args:
      physics: A `mujoco.Physics` instance.
      site_name: A string specifying the name of the target site.
      target_pos: A (3,) numpy array specifying the desired Cartesian position of
        the site, or None if the position should be unconstrained (default).
        One or both of `target_pos` or `target_quat` must be specified.
      target_quat: A (4,) numpy array specifying the desired orientation of the
        site as a quaternion, or None if the orientation should be unconstrained
        (default). One or both of `target_pos` or `target_quat` must be specified.
      joint_names: (optional) A list, tuple or numpy array specifying the names of
        one or more joints that can be manipulated in order to achieve the target
        site pose. If None (default), all joints may be manipulated.
      tol: (optional) Precision goal for `qpos` (the maximum value of `err_norm`
        in the stopping criterion).
      rot_weight: (optional) Determines the weight given to rotational error
        relative to translational error.
      regularization_threshold: (optional) L2 regularization will be used when
        inverting the Jacobian whilst `err_norm` is greater than this value.
      regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements.
      max_update_norm: (optional) The maximum L2 norm of the update applied to
        the joint positions on each iteration. The update vector will be scaled
        such that its magnitude never exceeds this value.
      progress_thresh: (optional) If `err_norm` divided by the magnitude of the
        joint position update is greater than this value then the optimization
        will terminate prematurely. This is a useful heuristic to avoid getting
        stuck in local minima.
      max_steps: (optional) The maximum number of iterations to perform.
      inplace: (optional) If True, `data` will be modified in place.
        Default value is False, i.e. a copy of `data` will be made.

    Returns:
      An `IKResult` namedtuple with the following fields:
        qpos: An (nq,) numpy array of joint positions.
        err_norm: A float, the weighted sum of L2 norms for the residual
          translational and rotational errors.
        steps: An int, the number of iterations that were performed.
        success: Boolean, True if we converged on a solution within `max_steps`,
          False otherwise.

    Raises:
      ValueError: If both `target_pos` and `target_quat` are None, or if
        `joint_names` has an invalid type.
    """

    dtype = data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        model = deepcopy(model)
        data = deepcopy(data)

    # Ensure that the Cartesian position of the site is up to date.
    mujoco.mj_fwdPosition(model, data)

    # Convert site name to index.
    site_id = model.site(site_name).id

    # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
    # update them in place, so we can avoid indexing overhead in the main loop.
    site_xpos = data.site(site_name).xpos
    site_xmat = data.site(site_name).xmat

    # This is an index into the rows of `update` and the columns of `jac`
    # that selects DOFs associated with joints that we are allowed to manipulate.
    if joint_names is None:
        dof_indices = slice(None)  # Update all DOFs.
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        # Find the indices of the DOFs belonging to each named joint. Note that
        # these are not necessarily the same as the joint IDs, since a single joint
        # may have >1 DOF (e.g. ball joints).
        indexer = named.model.dof_jntid.axes.row
        # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
        # indexer to map each joint name to the indices of its corresponding DOFs.
        dof_indices = indexer.convert_key_item(joint_names)
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    steps = 0
    success = False

    for steps in range(max_steps):

        err_norm = 0.0

        if target_pos is not None:
            # Translational error.
            err_pos[:] = target_pos - site_xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # Rotational error.
            mujoco.mju_mat2Quat(site_xquat, site_xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            logging.debug("Converged after %i steps: err_norm=%3g", steps, err_norm)
            success = True
            break
        else:
            # TODO(b/112141670): Generalize this to other entities besides sites.
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]

            # TODO(b/112141592): This does not take joint limits into consideration.
            reg_strength = (
                regularization_strength if err_norm > regularization_threshold else 0.0
            )
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength=reg_strength
            )

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                logging.debug(
                    "Step %2i: err_norm / update_norm (%3g) > "
                    "tolerance (%3g). Halting due to insufficient progress",
                    steps,
                    progress_criterion,
                    progress_thresh,
                )
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # Write the entries for the specified joints into the full `update_nv`
            # vector.
            update_nv[dof_indices] = update_joints

            # Update `qpos`, taking quaternions into account.
            mujoco.mj_integratePos(model, data.qpos, update_nv, 1)

            # Compute the new Cartesian position of the site.
            mujoco.mj_fwdPosition(model, data)

            logging.debug(
                "Step %2i: err_norm=%-10.3g update_norm=%-10.3g",
                steps,
                err_norm,
                update_norm,
            )

    if not success and steps == max_steps - 1:
        logging.warning(
            "Failed to converge after %i steps: err_norm=%3g", steps, err_norm
        )

    if not inplace:
        # Our temporary copy of data is about to go out of scope, and when
        # it does the underlying mjData pointer will be freed and data.qpos
        # will be a view onto a block of deallocated memory. We therefore need to
        # make a copy of data.qpos while data is still alive.
        qpos = data.qpos.copy()
    else:
        # If we're modifying data in place then it's fine to return a view.
        qpos = data.qpos

    return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.

    Args:
      jac_joints: The Jacobian of the end effector with respect to the joints. A
        numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
        and `nv` is the number of degrees of freedom.
      delta: The desired end-effector delta. A numpy array of shape `(3,)` or
        `(6,)` containing either position deltas, rotation deltas, or both.
      regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements. Default is zero, i.e. no regularization.

    Returns:
      An `(nv,)` numpy array of joint velocities.

    Reference:
      Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
      transpose, pseudoinverse and damped least squares methods.
      https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
