import numpy as np
import spatialmath as sm
from controllers.nullspace_ctrl import nullspace_ctrl
import mujoco
import mujoco.viewer
from pathlib import Path
from simulation.mujoco_helpers import qpos_from_site_pose
import time
import matplotlib

matplotlib.use("tkagg")


def simulate(model, data, duration, X_d, K):
    V_d = sm.Twist3()  # desired reference body twist is 0

    t_vec = np.arange(0, duration, model.opt.timestep)
    x_e = np.zeros((t_vec.size, 6))
    f_e = np.zeros((t_vec.size, 6))
    x = np.zeros((t_vec.size, 3))
    Q = np.zeros((t_vec.size, 7))
    # launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after the simulation duration expires
        for i, t in enumerate(t_vec):
            step_start = time.time()
            x[i, :] = data.site("panda_tool_center_point").xpos
            Q[i, :] = data.qpos[0:7]

            # advances simulation by all non-control dependent quantities
            mujoco.mj_step1(model, data)

            if t > 0.2:
                u = 1

            # evaluate the controller to get the control torques, as well as pose error
            # and control wrenches for introspection
            tau, x_e[i, :], f_e[i, :] = nullspace_ctrl(model, data, X_d, V_d, K)

            # set the MuJoCo controls according to the computed torques
            data.ctrl[0:7] = tau

            # actuate the finger ctrl to keep the gripper closed (a single force acts
            # on both fingers)
            data.ctrl[7] = -100.0

            # advance simulation fully, including the control torques
            mujoco.mj_step2(model, data)

            # update the viewer
            viewer.sync()

            # Rudimentary real-time keeping for visualization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Plotting
    _, axs = matplotlib.pyplot.subplots(2, 2)
    axs[0, 0].plot(t_vec, x_e[:, 0:3], label=["e_t_x", "e_t_y", "e_t_z"])
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].set_title("Translation Error")
    axs[0, 1].plot(t_vec, f_e[:, 0:3], label=["f_e_x", "f_e_y", "f_e_z"])
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_title("Cartesian Control Forces")
    axs[1, 0].plot(t_vec, x_e[:, 3:6], label=["e_r_x", "e_r_y", "e_r_z"])
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].set_title("Rotation Error")
    axs[1, 1].plot(t_vec, f_e[:, 3:6], label=["m_e_x", "m_e_y", "m_e_z"])
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_title("Cartesian Control Torques")

    axs[0, 0].set(ylabel="e_t [m]")
    axs[1, 0].set(ylabel="e_r [rad]")
    axs[0, 1].set(ylabel="f_e [N]")
    axs[1, 1].set(ylabel="m_e [Nm]")

    axs[1, 0].set(xlabel="t[s]")
    axs[1, 1].set(xlabel="t[s]")

    # maximize plot window
    mng = matplotlib.pyplot.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # Plot the end-effector path in a separate figure
    matplotlib.pyplot.figure()
    ax2 = matplotlib.pyplot.axes(projection="3d")
    matplotlib.pyplot.plot(x[:, 0], x[:, 1], x[:, 2])
    matplotlib.pyplot.plot(X_d.t[0], X_d.t[1], X_d.t[2], "r+")
    ax2.set(ylabel="y")
    ax2.set(xlabel="x")
    ax2.set(zlabel="z")
    ax2.axis("equal")
    ax2.set_title("End-effector Path")

    matplotlib.pyplot.figure()
    ax3 = matplotlib.pyplot.gca()
    matplotlib.pyplot.plot(t_vec, Q, label=["q1", "q2", "q3", "q4", "q5", "q6", "q7"])
    ax3.set(xlabel="t[s]")
    ax3.set(ylabel="q[rad]")
    ax3.set_title("Joint Angles")

    print(f"last q: {Q[-1,:]}")

    matplotlib.pyplot.show()


if __name__ == "__main__":
    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

    # Specify a desired diagonal stiffness matrix with translational (k_t) and
    # rotational (k_r) elements
    k_t = 500.0
    k_r = 50.0
    K = np.diag(np.hstack((np.ones(3) * k_t, np.ones(3) * k_r)))

    # control & simulation timestep
    timestep = 0.005

    # simulation duration
    duration = 150.0

    # load a model of the Panda manipulator
    xml_path = (
        str(Path(__file__).parent.resolve())
        + "/simulation/franka_emika_panda/panda.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    # The MuJoCo data instance is updated during the simulation. It's the central
    # element which stores all relevant variables
    data = mujoco.MjData(model)

    data.qpos = model.key("reset_config").qpos
    model.opt.timestep = timestep

    # compute forward dynamcis to update kinematic quantities
    mujoco.mj_forward(model, data)

    # design a reference goal pose
    X_d = sm.SE3.Rx(np.pi, t=np.array([0.5, 0, 0.4]))

    # corresponding alternate ns configuration
    q_ns_perturbed = np.array(
        [
            -3.67374456,
            -2.04725399,
            -0.36397499,
            -1.96872481,
            4.40004752,
            0.33883277,
            1.66728173,
            1.90605429e00,
            7.85301813e-01,
        ]
    )

    # find and set the joint configuration for the reference pose using IK
    res = qpos_from_site_pose(
        model,
        data,
        "panda_tool_center_point",
        target_pos=X_d.t,
        target_quat=sm.base.smb.r2q(X_d.R),
    )
    if not res.success:
        raise RuntimeError("[CartesianImpedanceControl]: IK did not converge.")

    data.qpos = q_ns_perturbed

    # run control & sim loop
    simulate(model, data, duration, X_d, K)
# 0.181537093
