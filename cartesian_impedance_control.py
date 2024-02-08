import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from swift import Swift
import spatialgeometry as sg
from controllers.cimp_simple import cimp_simple
import mujoco
from mujoco import viewer
from pathlib import Path
from simulation.mujoco_helpers import mjc_qpos_idx

def simulate(robot, model, data, duration, X_d):

    viewer = viewer.launch_passive(model, data)


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

    viewer.close()


if __name__ == "__main__":
    # control & simulation timestep
    timestep = 0.005

    # simulation duration
    duration = 5.0

    # pre-defined reset configuration
    q_d = np.array([0., -0.3, 0., -2.2, 0., 2.,  0.78539816])


    xml_path = str(Path(__file__).parent.resolve())+"/simulation/franka_emika_panda/panda.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[0:7] = q_d
    model.opt.timestep=timestep





    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

    # Make a Panda robot - unfortunately, the dynamics methods only seem to work on the
    # Denavit-Hartenberg parametrized models, whereas the Swift visualization only
    # works with models from a URDF. That's why we maintain two instances of the same
    # robot.
    robot = rtb.models.DH.Panda()
    robot_vis = rtb.models.Panda()


    X_d = robot.fkine(q_d, endlink="panda_link7")  # reference goal pose for the controller
    # data.site('panda_tool_center_point')

    mujoco.mj_forward(model,data)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site('panda_tool_center_point').id)

    jac_mj = np.vstack((jacp[:, 0:7], jacr[:, 0:7])) #
    R_ew = data.site('panda_tool_center_point').xmat.reshape(3,3).transpose()
    R_EW = np.zeros((6,6))
    R_EW[0:3 ,0:3] = R_ew
    R_EW[3:6, 3:6] = R_ew
    jac_mj_e = R_EW @ jac_mj # Body jacobian in ee frame

    jac_rt=robot_vis.jacobe(q_d)

    qM_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, qM_full, data.qM)# full invertia matrix, could use mj_mulM

    # Perturb the goal pose
    X = X_d * sm.SE3.Rz(np.pi/4, t=[0.1, 0.1, 0.1])

    # find joint configuration for perturbed pose
    q = robot.ikine_LM(X, q0=q_d).q

    # Set the joint coordinates to the perturbed q
    robot_vis.q = q
    robot.q = q




    # run control & sim loop
    simulate(robot, model, data, duration, X_d)

