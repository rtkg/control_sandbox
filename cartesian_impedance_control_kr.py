import numpy as np
from scipy.linalg import sqrtm
import spatialmath as sm
from controllers.cimp_simple import cimp_simple
import mujoco
import mujoco.viewer
from pathlib import Path
from simulation.mujoco_helpers import qpos_from_site_pose
import time
import matplotlib

# matplotlib.use("tkagg")

def simulate(model, data, duration, X_d, K, B, f_c, plotting):
    t = 0.0
    V_d = sm.Twist3()  # desired reference body twist is 0
    
    nullspace_conf = data.qpos

    t_vec = np.arange(0, duration, model.opt.timestep)
    x_e = np.zeros((t_vec.size, 6))
    f_e = np.zeros((t_vec.size, 6))
    x = np.zeros((t_vec.size, 3))
    
    NS_PER_SEC = 1000000000
    period_ns = 5000000
    nskews = 0
    # launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after the simulation duration expires
        tick_ns = ((time.time_ns() // NS_PER_SEC) + 1) * NS_PER_SEC
        sleep_period = (tick_ns - time.time_ns()) / NS_PER_SEC

        time.sleep(sleep_period)
        
        for i, t in enumerate(t_vec):
            step_start = time.time()
            x[i, :] = data.site("tool_center_point").xpos

            # advances simulation by all non-control dependent quantities
            mujoco.mj_step1(model, data)

            # evaluate the controller to get the control torques, as well as pose error
            # and control wrenches for introspection
            tau, x_e[i, :], f_e[i, :] = cimp_simple(model, data, X_d, V_d, K, B, f_c, n_conf=nullspace_conf, tcp_site_id="tool_center_point")

            # set the MuJoCo controls according to the computed torques
            data.ctrl[0:7] = tau

            # actuate the finger ctrl to keep the gripper closed (a single force acts
            # on both fingers)
            # ! Not provided with the KR model yet
            # data.ctrl[7] = -100.0

            # advance simulation fully, including the control torques
            mujoco.mj_step2(model, data)
            
            # update the viewer
            viewer.sync()

            # real-time alignments
            tick_ns = tick_ns + period_ns
            sleep_period_ns = tick_ns - time.time_ns()
            sleep_period = (sleep_period_ns) / NS_PER_SEC
            #print(sleep_period_ns)
            if sleep_period >= 0:
                time.sleep(sleep_period)
            else:
                nper_skipped = (-sleep_period_ns // period_ns) + 1
                tick_ns += (nper_skipped * period_ns)
                nskews += nper_skipped
                print(f"time skew occured! (periods skipped: {nper_skipped})")
                time.sleep((tick_ns - time.time_ns()) / NS_PER_SEC)

            # Rudimentary real-time keeping for visualization
            #time_until_next_step = model.opt.timestep - (time.time() - step_start)
            #if time_until_next_step > 0:
            #    time.sleep(time_until_next_step)
            #else:
            #    nrcounter += 1

    # Plotting (not working with the mjpython on OSX)
    if (plotting):
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

        matplotlib.pyplot.show()
    

if __name__ == "__main__":
    np.set_printoptions(precision=20) 
    # This script runs a simple control & simulation loop where the arm end-effector
    # is perturbed by a given transformation

    # Specify a desired diagonal stiffness matrix with translational (k_t) and
    # rotational (k_r) elements
    k_t = 5000.0
    k_r = 50.0
    # K = np.diag(np.hstack((np.ones(3) * k_t, np.ones(3) * k_r)))
    # Martin's test case
    #K = np.diag(np.hstack((np.ones(3) * k_t, np.array([10.0, 10.0, 10.0]))))
    K = np.diag(np.hstack((np.array([100.0, 100.0, 100.0]), np.array([10.0, 10.0, 10.0]))))

    # Damping matrix entry
    B = 2 * sqrtm(K)  # critical damping assuming unit mass
    if 1:
        B[0, 0] = 10.0
        B[1, 1] = 10.0
        B[2, 2] = 10.0
        B[3, 3] = 6.32455532 
        B[4, 4] = 6.32455532
        B[5, 5] = 6.32455532

    # Target compliance (if user wants robot to generate force/torque along given axes)
    f_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # control & simulation timestep
    timestep = 0.005

    # simulation duration
    duration = 1500.0

    # load a model of the Panda manipulator
    xml_path = (
        str(Path(__file__).parent.resolve())
        + "/simulation/kassow_810/scene.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    # The MuJoCo data instance is updated during the simulation. It's the central
    # element which stores all relevant variables
    data = mujoco.MjData(model)
    data.qpos = model.key('nullspace_config').qpos
    model.opt.timestep = timestep

    # compute forward dynamcis to update kinematic quantities
    mujoco.mj_forward(model, data)

    # get actual tcp
    quat = sm.base.smb.r2q(data.site("tool_center_point").xmat.reshape(3, 3))
    quat = quat / np.linalg.norm(quat)
    X = sm.SE3.Rt(sm.base.smb.q2r(quat), data.site("tool_center_point").xpos)

    # design a pertubation expressed in the end-effector frame
    # - no initial perturbation used here    
    X_p = sm.SE3.Rx(0, t=np.array([0.0, 0.0, 0.0]))

    # design a reference goal pose
    X_d = X * X_p

    # run control & sim loop
    simulate(model, data, duration, X_d, K, B, f_c, False)
