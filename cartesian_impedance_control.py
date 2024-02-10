import numpy as np
import spatialmath as sm
from controllers.cimp_simple import cimp_simple
import mujoco
import mujoco.viewer
from pathlib import Path
from simulation.mujoco_helpers import qpos_from_site_pose
import time
import matplotlib.pyplot as plt


def simulate(model, data, duration, X_d, K):
    t = 0.0
    V_d = sm.Twist3()  # desired reference body twist is 0

    t_vec = np.arange(0, duration, model.opt.timestep)
    x_e = np.zeros((t_vec.size, 6))
    # launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Close the viewer automatically after the simulation duration expires

        for i, t in enumerate(t_vec):
            step_start = time.time()

            # advances simulation by all non-control dependent quantities
            mujoco.mj_step1(model, data)

            # evaluate the controller to get the control torques
            tau, x_e[i, :] = cimp_simple(model, data, X_d, V_d, K)

            # set the MuJoCo controls according to the computed torques
            data.ctrl[0:7] = tau

            # actuate the finger ctrl to keep closed
            data.ctrl[7] = -100.0

            # advance simulation fully
            mujoco.mj_step2(model, data)

            viewer.sync()

            # Rudimentary real time keeping for visualization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t_vec, x_e[:, 0:3], label=["e_t_x", "e_t_y", "e_t_z"])
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].set_title('Translation Error')
    axs[0, 1].plot(t_vec, np.linalg.norm(x_e[:, 0:3], axis=1), label="||e_t||")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_title('Translation Error Norm')
    axs[1, 0].plot(t_vec, x_e[:, 3:6], label=["e_r_x", "e_r_y", "e_r_z"])
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].set_title('Rotation Error')
    axs[1, 1].plot(t_vec, np.linalg.norm(x_e[:, 3:6], axis=1), label="||e_r||")
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_title('Rotation Error Norm')

    axs[0, 0].set(ylabel='e_t [m]')
    axs[1, 0].set(ylabel='e_r [rad]')
    axs[1, 0].set(xlabel='t[s]')
    axs[1, 1].set(xlabel='t[s]')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

    # Specify a desired diagonal stiffness matrix with translational (k_t) and
    # rotational (k_r) elements
    k_t = 500.0
    k_r = 50
    K = np.diag(np.hstack((np.ones(3)*k_t, np.ones(3)*k_r)))

    # control & simulation timestep
    timestep = 0.005

    # simulation duration
    duration = 5.0

    # pre-defined reference configuration
    q_d = np.array([0., -0.3, 0., -2.2, 0., 2.,  0.78539816])

    # load a model of the Panda manipulator
    xml_path = str(Path(__file__).parent.resolve())+"/simulation/franka_emika_panda/panda.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[0:7] = q_d
    model.opt.timestep = timestep

    # compute forward dynamcis to update kinematic quantities
    mujoco.mj_forward(model, data)

    # get the reference goal pose
    X_d = sm.SE3.Rt(data.site('panda_tool_center_point').xmat.reshape(3, 3), data.site('panda_tool_center_point').xpos)

    # get a perturbed pose
    X = X_d * sm.SE3.Rz(np.pi / 4, t=[0.1, 0.1, 0.1])

    # find and set the joint configuration for the perturbed pose using IK
    res = qpos_from_site_pose(model, data, "panda_tool_center_point", target_pos=X.t, target_quat=sm.base.smb.r2q(X.R))
    data.qpos = res.qpos

    # run control & sim loop
    simulate(model, data, duration, X_d, K)
