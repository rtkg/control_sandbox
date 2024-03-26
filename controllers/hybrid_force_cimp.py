from scipy.linalg import sqrtm
from simulation.mujoco_helpers import mjc_body_jacobian, mjc_body_jacobian_derivative
import spatialmath as sm
import numpy as np
from copy import copy
from helpers.damped_pinv import damped_pinv
from helpers.ec_reparametrization import ec_reparametrization
import mujoco


def hybrid_force_cimp(
    model, data, X_d, V_d, dV_d, K, A, f, stiffness_frame="reference"
):
    """
    Hybrid force Impedance Controller formulated according to the algorithm in [1],
    p. 441, Eq. (11.61) without full arm dynamcis compensation (only gravitational load
    and centripetal / coriolis forces are compensated) and a virtual mass of M=0. Approximate
    critical damping is computed automatically in dependence of the given stiffness under a
    unit mass assumption.

    Args:
        X_d (SE3) ... Desired end-effector pose expressed in the base frame
        V_d (Twist3) .. Desired end-effector body twist expressed in the end-effector
                        frame
        dV_d (Twist3) .. Desired end-effector body acceleration expressed in the end-effector
                        frame
        K (np.ndarray(6 , 6)) ... Desired stiffness
        A (np.ndarray(k, 6)) ... Pfaffian constraint matrix such that A * V = 0
        f (np.ndarray(6)) ... Arbitrary desired contact wrench
        stiffness_frame (string) ... "reference" or "end_effector" - specifies in which frame
                                     K, A, and f are expressed

    Returns:
        tau (np.ndarray) ... joint control torquesmjc_world_jacobian_derivative
        x_e (np.ndarray) ... pose error
        f_u  (np.ndarray) ... generated cartesian control force

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

    Jdot = mjc_body_jacobian_derivative(model, data)  # Jacobian derivative
    V = J @ dq  # current ee body twist

    # =============== Cartesian wrench-resolved control law ==================
    # the control law is formulated w.r.t. the ee-frame if K, A, and f are expressed in the reference
    # frame, they need to be transformed

    B = 2 * sqrtm(K)  # critical damping approximation assuming unit mass
    X_e = X.inv() * X_d  # error pose (i.e., the ref frame expressed in the  ee frame)
    Ad_e = X_e.Ad()  # adjoint of the error pose

    # if stiffness_frame == "end_effector":
    #     S_M = np.eye(6) - A
    #     S_F = copy(A)
    #     E_f = copy(f)
    #     E_K = copy(K)
    #     E_B = copy(B)
    # elif stiffness_frame == "reference":
    #     S_M = Ad_e @ (np.eye(6) - A) @ np.linalg.pinv(Ad_e)
    #     S_F = np.linalg.pinv(Ad_e.transpose()) @ A @ Ad_e.transpose()
    #     E_f = np.linalg.pinv(Ad_e).transpose() @ f
    #     E_K = Ad_e @ K @ np.linalg.pinv(Ad_e)
    #     E_B = Ad_e @ B @ np.linalg.pinv(Ad_e)
    # else:
    #     raise ValueError("Invalid stiffness_frame")

    # motion quantity errors in exponential coordinates
    # ee body twist expressed in the reference frame is transformed to the current ee frame
    # using the Adjoint
    x_e = X_e.norm().log(twist="true")  # pose error in exponential coordinates
    v_e = Ad_e @ V_d - V  # twist error
    # cross product representation of the twist error, i.e., the "small adjoint"
    ad_e = np.zeros((6, 6))
    ad_e[0:3, 0:3] = sm.base.smb.skew(v_e[3:6])
    ad_e[0:3, 3:6] = sm.base.smb.skew(v_e[0:3])
    ad_e[3:6, 3:6] = sm.base.smb.skew(v_e[3:6])

    a_ff = Ad_e @ dV_d + ad_e @ V_d  # feedforward acceleration

    # alternate error formulation where only rotation is parametrized by 3d
    # exponential coordinates ([1], p. 420, Eq. (11.18)). This leads to
    # straight-line motions in position space.
    if 1:
        x_e[0:3] = X_e.t

    # task-space manipulator inertia
    qM = np.zeros((7, 7))
    mujoco.mj_fullM(model, qM, data.qM)
    xM = np.linalg.inv(J @ np.linalg.inv(qM[0:7, 0:7]) @ J.transpose())

    # compute the control wrench, force control is feedforward only ([1], p. 441, eq. (11.61))
    # f_u = P @ xM @ (B @ v_e + K @ x_e) + S_F @ E_f
    f_u = xM @ (B @ v_e + K @ x_e + a_ff)

    # Mapping the external control wrench to joint torques and compensating
    # the manipulator dynamics (gravity + coriolis / centripetal only).
    tau = data.qfrc_bias[0:7] + J.transpose() @ (f_u - xM @ Jdot @ dq)

    # add the nullspace torques which will bias the manipulator to the desired
    # nullspace bias configuration
    q_ns = model.key("nullspace_config").qpos[0:7]
    K_ns = np.eye(7) * 5
    B_ns = 2 * sqrtm(K_ns)
    # compute joint accelerations in the nullspace of the end-effector accelerations
    ddq_ns = (np.eye(7) - damped_pinv(J, 1e-5) @ J) @ (K_ns @ (q_ns - q) - B_ns @ dq)
    # map nullspace accelerations to torques
    tau_ns = qM[0:7, 0:7] @ ddq_ns

    # return x_e and f_u for introspection, in addition to the control torques
    return tau + tau_ns, x_e, f_u
