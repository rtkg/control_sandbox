from scipy.linalg import sqrtm
from numpy.linalg import norm


def cimp_simple(robot, X_d, V_d, K):

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

    # just for printing
    err = X_e.log(twist="true")
    print(f"Translational error: {norm(err[0:3])}, rotational error: {norm(err[3:6])}")

    # Mapping the external control force to joint torques and adding the gravitational
    # load torques
    tau = robot.gravload(q) + J.transpose() @ f_e
    return tau
