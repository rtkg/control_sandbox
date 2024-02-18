from copy import copy
import numpy as np


def ec_reparametrization(x_r):
    """
    dynamic reparametrization of the exponential coordinate representation of orientation in the vicinity of phi = pi
    according to https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
    """

    x_r = copy(x_r)

    phi = np.linalg.norm(x_r)
    if np.abs(phi - np.pi) < 1e-3:
        x_r = (1 - 2 * np.pi / phi) * x_r

    return x_r
