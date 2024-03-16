import numpy as np
from copy import copy


def generate_motion_profile(q_0, q_f, t):
    """
    Analytically computes a minimum-jerk motion profile (q(t), dq(t), ddq(t)).
    See, e.g.,
    https://mika-s.github.io/python/control-theory/trajectory-generation/2017/12/06/trajectory-generation-with-a-minimum-jerk-trajectory.html
    Args:
        q_0(array-like): n x 1 initial positions in n dimensions
        q_f(array-like): n x 1 target positions in n dimensions
        t(array-like): m x 1 m-dimensional time vector
    Returns:
        q(np.ndarray) mxn position trajectory
        dq(np.ndarray): mxn velocity trajectory
        ddq(np.ndarray): mxn acceleration trajectory
        dddq(np.ndarray): mxn jerk trajectory
    """

    if not isinstance(q_0, np.ndarray):
        q_0 = np.asarray(q_0)

    if not isinstance(q_f, np.ndarray):
        q_f = np.asarray(q_f)

    # shift the time vector to start at 0 (underlying assumption of subsequent computations)
    t_ = copy(np.asarray(t))
    t_ -= t_[0]

    T = t_[-1]

    q = (
        np.outer(10 * (t_ / T) ** 3 - 15 * (t_ / T) ** 4 + 6 * (t_ / T) ** 5, q_f - q_0)
        + q_0
    )
    dq = np.outer(
        30 * (t_ / T) ** 2 - 60 * (t_ / T) ** 3 + 30 * (t_ / T) ** 4, (q_f - q_0) / T
    )
    ddq = np.outer(
        60 * t_ / T - 180 * (t_ / T) ** 2 + 120 * (t_ / T) ** 3, (q_f - q_0) / T**2
    )
    dddq = np.outer(60 * 1 / T - 360 * t_ / T + 360 * (t_ / T) ** 2, (q_f - q_0) / T**3)

    return q, dq, ddq, dddq
