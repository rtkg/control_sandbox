import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from swift import Swift
import spatialgeometry as sg
from controllers.cimp_simple import cimp_simple
import mujoco
from mujoco import viewer
from pathlib import Path
from simulation.mujoco_helpers import mjc_body_jacobian, qpos_from_site_pose

def simulate(model, data, duration, X_d):

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
    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

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

    # compute forward dynamcis to update kinematic quantities
    mujoco.mj_forward(model, data)

    X_d = sm.SE3.Rt(data.site('panda_tool_center_point').xmat.reshape(3, 3), data.site('panda_tool_center_point').xpos)

    # Perturb the goal pose
    X = X_d * sm.SE3.Rz(np.pi/4, t=[0.1, 0.1, 0.1])

    # find and set the joint configuration for the perturbed pose using IK
    res = qpos_from_site_pose(model, data, "panda_tool_center_point", target_pos=X.t, target_quat=sm.base.smb.r2q(X.R))
    data.qpos = res.qpos

    # compute forward dynamcis to update kinematic quantities
    mujoco.mj_forward(model, data)

    #qM_full = np.zeros((model.nv, model.nv))
    #mujoco.mj_fullM(model, qM_full, data.qM)# full invertia matrix, could use mj_mulM

    # run control & sim loop
    simulate(model, data, duration, X_d)

