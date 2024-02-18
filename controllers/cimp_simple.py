from scipy.linalg import sqrtm
from simulation.mujoco_helpers import mjc_body_jacobian, mjc_body_jacobian_derivative
import spatialmath as sm
import numpy as np
from helpers.damped_pinv import damped_pinv
from helpers.ec_reparametrization import ec_reparametrization
import mujoco


def cimp_simple(model, data, X_d, V_d, K):
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
    V = J @ dq  # current ee body twist
    X_e = X.inv() * X_d  # error pose

    # simple impedance control law in exponential coordinates. The desired ee body
    # twist expressed in the reference frame is transformed to the current ee frame
    # using the Adjoint
    B = 2 * sqrtm(K)  # critical damping assuming unit mass
    x_e = X_e.norm().log(twist="true")  # pose error in exponential coordinates
    v_e = X_e.Ad() @ V_d - V  # twist error

    # dynamic reparametrization of the orientation error in the vicinity of e_phi = pi
    # according to https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
    if 0:
        x_e[3:6] = ec_reparametrization(x_e[3:6])

    # alternate error formulation where only rotation is parametrized by 3d
    # exponential coordinates ([1], p. 420, Eq. (11.18)). This leads to
    # straight-line motions in position space.
    if 1:
        x_e[0:3] = X_e.t
        L = np.eye(6)
        L[3:6, 3:6] = X_e.R
        v_e = L @ V_d - V

    # Computing the external control wrench
    f_e = B @ v_e + K @ x_e

    # Jacobian derivative
    Jdot = mjc_body_jacobian_derivative(model, data)

    # task-space manipulator inertia
    qM = np.zeros((9, 9))
    mujoco.mj_fullM(model, qM, data.qM)
    xM = np.linalg.pinv(J @ np.linalg.pinv(qM[0:7, 0:7]) @ J.transpose())

    # Mapping the external control wrench to joint torques and compensating
    # the manipulator dynamics (gravity + coriolis / centripetal only).
    # Here, the control wrenches are interpreted as accelerations and scaled by the
    # configuraiton-dependent task-space inertia matrix to bring them to wrench-space
    # (cf. [1], p. 434. Eq. (11.47)). The contribution of Jdot * dq is minor and could
    # be omitted if computational effort is of concern
    tau = data.qfrc_bias[0:7] + J.transpose() @ xM @ (f_e - Jdot @ data.qvel[0:7])

    # add the nullspace torques which will bias the manipulator to the desired
    # nullspace bias configuration
    q_ns = model.key("nullspace_config").qpos[0:7]
    K_ns = np.eye(7) * 0.1
    B_ns = 2 * sqrtm(K_ns)
    tau_ns = (np.eye(7) - damped_pinv(J, 1e-5) @ J) @ (K_ns @ (q_ns - q) - B_ns @ dq)

    # return x_e and f_e for introspection, in addition to the control torques
    return tau + tau_ns, x_e, f_e
