import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from swift import Swift
import spatialgeometry as sg
from controllers.cimp_simple import cimp_simple


def simulate(robot, robot_vis, env, period, duration, X_d):

    # end-effector axes for visualization
    ee_axes = sg.Axes(0.1)

    # goal reference axes for visualization
    ref_axes = sg.Axes(0.1)

    # Add the axes to the environment
    env.add(ee_axes)
    env.add(ref_axes)

    # Set the reference axes to the desired ee pose
    ref_axes.T = X_d

    # Specify a desired diagonal stiffness matrix with translational (k_t) and
    # rotational (k_r) elements
    k_t = 500.0
    k_r = 0.2
    K = np.diag(np.hstack((np.ones(3)*k_t, np.ones(3)*k_r)))
    t = 0.0
    V_d = sm.Twist3()  # desired reference body twist is 0
    while t < duration:
        q = robot.q
        qd = robot.qd

        # evaluate the controller to get the control torques
        tau = cimp_simple(robot, X_d, V_d, K)

        # compute the forward dynamics
        qdd = np.linalg.inv(robot.inertia(q)) @ (tau - robot.coriolis(q, qd) @ qd-robot.gravload(q))

        # update the robot kinematics using simple euler integration
        robot.q += period * qd
        robot.qd += period * qdd
        robot_vis.q = robot.q

        # update the visualization
        ee_axes.T = robot.fkine(robot.q)
        env.step(period)

        t += period  # increase time


if __name__ == "__main__":
    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

    # Make a Panda robot - unfortunately, the dynamics methods only seem to work on the
    # Denavit-Hartenberg parametrized models, whereas the Swift visualization only
    # works with models from a URDF. That's why we maintain two instances of the same
    # robot.
    robot = rtb.models.DH.Panda()
    robot_vis = rtb.models.Panda()

    q_d = robot_vis.qr  # pre-defined reset configuration
    X_d = robot.fkine(q_d)  # reference goal pose for the controller

    # Perturb the goal pose
    X = X_d * sm.SE3.Rz(np.pi/4, t=[0.1, 0.1, 0.1])

    # find joint configuration for perturbed pose
    q = robot.ikine_LM(X, q0=q_d).q

    # Set the joint coordinates to the perturbed q
    robot_vis.q = q
    robot.q = q

    # Make the environment
    env = Swift()

    # Launch the visualization, will open a browser tab in your default
    # browser (chrome is recommended)
    # The realtime flag will ask the visualization to display as close as
    # possible to realtime as apposed to as fast as possible
    env.launch(realtime=True)

    env.add(robot_vis)
    env.step(0.0)  # update visualization

    # control & simulation period
    period = 0.01

    # simulation duration
    duration = 5.0

    # run control & sim loop
    simulate(robot, robot_vis, env, period, duration, X_d)
