import numpy as np

from helpers.generate_motion_profile import generate_motion_profile
import spatialmath as sm


def generate_line_motion(X_0, timestep, traj_duration, n_traj=1):

    X_d = []
    V_d = []

    for i in range(n_traj):
        # minimum-jerk trajectory in R^3 going in positive y
        P, dP, _, _ = generate_motion_profile(
            np.zeros(3),
            np.array([0, 0.2, 0]),
            np.arange(0, traj_duration / 2.0, timestep),
        )
        for p, dp in zip(P, dP):
            X_d_local = sm.SE3.Trans(p)
            X_d.append(X_0 * X_d_local)
            # body twist
            V_d.append(sm.Twist3(np.array([dp[0], dp[1], dp[2], 0.0, 0.0, 0.0])))

        # minimum-jerk trajectory in R^3 going in negative y
        P, dP, _, _ = generate_motion_profile(
            np.array([0, 0.2, 0]),
            np.zeros(3),
            np.arange(0, traj_duration / 2.0, timestep),
        )
        for p, dp in zip(P, dP):
            X_d_local = sm.SE3.Trans(p)
            X_d.append(X_0 * X_d_local)
            # body twist
            V_d.append(sm.Twist3(np.array([dp[0], dp[1], dp[2], 0.0, 0.0, 0.0])))

    return X_d, V_d
