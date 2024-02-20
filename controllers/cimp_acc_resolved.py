from scipy.linalg import sqrtm
from simulation.mujoco_helpers import mjc_body_jacobian, mjc_body_jacobian_derivative
import spatialmath as sm
import numpy as np
from helpers.damped_pinv import damped_pinv
from helpers.ec_reparametrization import ec_reparametrization
import mujoco


def cimp_acc_resolved(model, data, X_d, V_d, K):
    """
        Simple Cartesian Impedance Controller formulated according to the algorithm in [1],
        p. 444, Eq. (11.65) without full arm dynamcis compensation (only gravitational load
        and centripetal / coriolis forces are compensated) and a virtual mass of M=0. Note,
        that this can be seen as a force controller (cf. [1], p. 435, Eq. (11.50)), where
        the desired external end-effector wrench is specified by a cartesian impedance
        tracking law. Critical damping is computed automatically in dependence of the given
        stiffness under a unit mass assumption.
    Jdot @ data.qvel[0:7]. MuJoCo data struct
            X_d (SE3) ... Desired end-effector pose expressed in the base frame
            V_d (Twist3) .. Desired end-effector body twist expressed in the end-effector
                            frame
            K (np.ndarray(6 , 6)) ... Desired stiffness expressed in the end-effector frame

        Returns:
            tau (np.ndarray) ... joint control torques
            x_e (np.ndarray) ... pose error
            f_e  (np.ndarray) ... generated cartesian control force

        [1] ... Lynch, K. M., & Park, F. C. (2017). Modern robotics. Cambridge University
                Press.
    """

    q = data.qpos[0:7]  # current joint positions
    dq = data.qvel[0:7]  # current joint velocities

    # current robot ee-pose expressed in the base frame, renormalize orientation
    # quaternion to avoid numerical issues downstream
    quat = sm.base.smb.r2q(data.site("panda_tool_center_point").xmat.reshape(3, 3))
    quat = quat / np.linalg.norm(quat)
    X = sm.SE3.Rt(sm.base.smb.q2r(quat), data.site("panda_tool_center_point").xpos)

    J = mjc_body_jacobian(model, data)  # ee body jacobian expressed in the ee frame
    # Jacobian derivative
    Jdot = mjc_body_jacobian_derivative(model, data)
    V = J @ dq  # current ee body twist
    X_e = X.inv() * X_d  # error pose

    # task-space manipulator inertia
    qM = np.zeros((9, 9))
    mujoco.mj_fullM(model, qM, data.qM)
    xM = np.linalg.pinv(J @ np.linalg.pinv(qM[0:7, 0:7]) @ J.transpose())

    # simple impedance control law in exponential coordinates. The desired ee body
    # twist expressed in the reference frame is transformed to the current ee frame
    # using the Adjoint
    B = 2 * sqrtm(xM) @ sqrtm(K)
    x_e = X_e.norm().log(twist="true")  # pose error in exponential coordinates
    v_e = X_e.Ad() @ V_d - V  # twist error

    # alternate error formulation where only rotation is parametrized by 3d
    # exponential coordinates ([1], p. 420, Eq. (11.18)). This leads to
    # straight-line motions in position space.
    if 1:
        x_e[0:3] = X_e.t
        L = np.eye(6)
        L[3:6, 3:6] = X_e.R
        v_e = L @ V_d - V

    # Computing the task-space control acceleration
    ddx_u = np.linalg.pinv(xM) @ (B @ v_e + K @ x_e)

    # mapping to joint-space control acceleration
    ddq_u = damped_pinv(J, 1e-5) @ (ddx_u - Jdot @ data.qvel[0:7])

    # add nullspace accelerations which will bias the manipulator to the desired
    # nullspace bias configuration
    q_ns = model.key("nullspace_config").qpos[0:7]
    K_ns = np.eye(7) * 5
    B_ns = 2 * sqrtm(K_ns)
    # compute joint accelerations in the nullspace of the end-effector accelerations
    ddq_ns = (np.eye(7) - damped_pinv(J, 1e-5) @ J) @ (K_ns @ (q_ns - q) - B_ns @ dq)
    ddq_u += ddq_ns

    # mapping to joint torques and adding dynamics bias compensation
    tau_u = qM[0:7, 0:7] @ ddq_u + data.qfrc_bias[0:7]

    # return x_e and ddx_u for introspection, in addition to the control torques
    return tau_u, x_e, ddx_u
