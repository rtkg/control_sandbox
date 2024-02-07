from scipy.linalg import sqrtm
from numpy.linalg import norm


def cimp_simple(robot, X_d, V_d, K):
    '''
    Simple Cartesian Impedance Controller formulated according to the algorithm in [1],
    p. 444, Eq. (11.65) without full arm dynamcis compensation (only gravitational load
    is compensated) and a virtual mass of M=0. Note, that this can be seen as a force
    controller (cf. [1], p. 435, Eq. (11.50)), where the desired external end-effector
    wrench is specified by a cartesian impedance tracking law. Critical damping is
    computed automatically in dependence of the given stiffness under a unit mass
    assumption.

    Args:
        robot (roboticstoolbox.models) ... controlled robot. It is assumed that the
                                           member variables robot.q and robot.dq are
                                           set.
        X_d (SE3) ... Desired end-effector pose expressed in the base frame
        V_d (Twist3) .. Desired end-effector body twist expressed in the end-effector
                        frame

    Returns:
        tau (np.ndarray) ... joint control torques

    [1] ... Lynch, K. M., & Park, F. C. (2017). Modern robotics. Cambridge University
            Press.
    '''

    q = robot.q  # current joint positions
    qd = robot.qd  # current joint velocities
    X = robot.fkine(q)  # current robot ee-pose expressed in the base frame
    J = robot.jacobe(q)  # ee body jacobian expressed in the ee frame
    V = J @ qd  # current ee body twist
    X_e = X.inv() * X_d  # error pose

    # simple impedance control law in exponential coordinates. The desired ee body
    # twist expressed in the reference frame is transformed to the current ee frame
    # using the Adjoint
    B = 2 * sqrtm(K)  # critical damping assuming unit mass
    f_e = B @ (X_e.Ad() @ V_d - V) + K @ X_e.log(twist="true")

    # just for introspection
    err = X_e.log(twist="true")
    print(f"Translational error: {norm(err[0:3])}, rotational error: {norm(err[3:6])}")

    # Mapping the external control force to joint torques and adding the gravitational
    # load torques
    tau = robot.gravload(q) + J.transpose() @ f_e
    return tau
