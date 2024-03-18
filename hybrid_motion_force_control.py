import numpy as np
import spatialmath as sm
from controllers.hybrid_force_cimp import hybrid_force_cimp
from helpers.generate_line_motion import generate_line_motion
import mujoco
import mujoco.viewer
from pathlib import Path
import time
import matplotlib


def simulate(
    model, data, X_d, V_d, K, A, f, stiffness_frame="reference", plotting=True
):
    duration = model.opt.timestep * len(X_d)

    t_vec = np.arange(0, duration, model.opt.timestep)
    x_e = np.zeros((t_vec.size, 6))
    f_e = np.zeros((t_vec.size, 6))
    x = np.zeros((t_vec.size, 3))
    x_d = np.zeros((t_vec.size, 3))
    # launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after the simulation duration expires
        for i, t in enumerate(t_vec):
            step_start = time.time()
            x[i, :] = data.site("panda_tool_center_point").xpos
            x_d[i, :] = X_d[i].t

            # advances simulation by all non-control dependent quantities
            mujoco.mj_step1(model, data)

            # evaluate the controller to get the control torques, as well as pose error
            # and control wrenches for introspection
            tau, x_e[i, :], f_e[i, :] = hybrid_force_cimp(
                model, data, X_d[i], V_d[i], K, A, f, stiffness_frame
            )

            # set the MuJoCo controls according to the computed torques
            data.ctrl[0:7] = tau

            # advance simulation fully, including the control torques
            mujoco.mj_step2(model, data)

            # update the viewer
            viewer.sync()

            # Rudimentary real-time keeping for visualization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Plotting
    if plotting:
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

        # Plot the end-effector path in a separate figure
        matplotlib.pyplot.figure()
        ax2 = matplotlib.pyplot.axes(projection="3d")
        matplotlib.pyplot.plot(x[:, 0], x[:, 1], x[:, 2], label="x")
        matplotlib.pyplot.plot(x_d[:, 0], x_d[:, 1], x_d[:, 2], label="x_d")
        ax2.set(ylabel="y")
        ax2.set(xlabel="x")
        ax2.set(zlabel="z")
        ax2.axis("equal")
        ax2.set_title("End-effector Path")

        matplotlib.pyplot.show()


if __name__ == "__main__":
    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

    # Specify a desired diagonal stiffness matrix with translational (k_t) and
    # rotational (k_r) elements
    k_t = 500.0
    k_r = 50.0
    K = np.diag(np.hstack((np.ones(3) * k_t, np.ones(3) * k_r)))

    # Specify a desired contact wrench
    f = np.array([0, 0, 10.0, 0, 0, 0])

    # Specify a desired Pfaffian constraint matrix A (see Lynch textbook (https://hades.mech.northwestern.edu/images/7/7f/MR.pdf), pp. 439)
    # This is a k x 6 matrix, where k is the number of end-effector twist constraints, i.e., A * V = 0. In the context of hybrid
    # force/motion control, this means that the end-effector is free to move in 6-k directions, and constrained (i.e., force-controlled) in k directions.

    A = np.zeros((6, 6))
    A[2, 2] = 1

    # control & simulation timestep
    timestep = 0.005

    # trajectory duration
    duration = 5.0

    # optionally plot various simulation variables for introspection
    plotting = True

    # "reference" or "end_effector" - specifies in which frame K, A, and f are expressed
    stiffness_frame = "end_effector"

    # load a model of the Panda manipulator
    xml_path = (
        str(Path(__file__).parent.resolve())
        + "/simulation/franka_emika_panda/panda_obstacle.xml"
    )

    model = mujoco.MjModel.from_xml_path(xml_path)
    # The MuJoCo data instance is updated during the simulation. It's the central
    # element which stores all relevant variables
    data = mujoco.MjData(model)
    data.qpos = model.key("reset_config").qpos
    model.opt.timestep = timestep

    # compute forward dynamcis to update kinematic quantities
    mujoco.mj_forward(model, data)

    # current ee pose- xM @ Jdot @ data.qvel[0:7]
    X_0 = sm.SE3.Rt(
        data.site("panda_tool_center_point").xmat.reshape(3, 3),
        data.site("panda_tool_center_point").xpos,
        check=False,
    )

    # a test reference trajectory consisting simply of a constant setpoint with zero desired twist
    # X_d = [X_0] * int(duration / timestep)
    # V_d = [sm.Twist3()] * int(duration / timestep)

    # a test reference trajectory describing a back-and-forth motion along one axis
    X_d, V_d = generate_line_motion(X_0, timestep, duration, 20)

    # run control & sim loop
    simulate(model, data, X_d, V_d, K, A, f, stiffness_frame, plotting)
