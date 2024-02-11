import numpy as np


def damped_pinv(J, z):
    '''
    Computes the damped Pseudoinverse according to
    J_dagger = J^T(JJ^T-z^2*I)-1

    Args:
        J (np.ndarray) ... Matrix to invert
        z (float)      ... scalar damping factor >= 0
    '''
    n = J.shape[0]

    return J.transpose() @ np.linalg.pinv(J @ J.transpose() - z**2 * np.eye(n))
