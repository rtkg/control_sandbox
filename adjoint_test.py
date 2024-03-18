import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

X = sm.SE3()
X_d = sm.SE3.Rz(-np.pi / 2, t=np.array([0.5, 0.5, 0.0]))
X_e = X.inv() * X_d  # error pose
K_d = np.diag([10, 10, 10, 1, 1, 1])
Ad = X_e.Ad()

# twist constraint matrix
A = np.zeros((6, 6))
A[1, 1] = 1  # constrain D_v_y = 0, i.e., D_y is force-controlled
f = np.array([0, 1, 0, 0, 0, 0])

print("Adjoint: \n", np.round(Ad, decimals=2))

print("A: \n", A)

print(
    "Ad @ (I-A) @ Ad^{-1}: \n",
    np.round(Ad @ (np.eye(6) - A) @ np.linalg.pinv(Ad), decimals=2),
)

print(
    "Ad^{-T} @ (A) @ Ad^T: \n",
    np.round(np.linalg.inv(Ad.transpose()) @ A @ Ad.transpose(), decimals=2),
)


R = copy(Ad)
R[0:3, 3:6] = np.zeros((3, 3))
K_d = np.diag([1, 2, 3, 4, 5, 6])
print("K_d: \n", K_d)
print("K: \n ", np.round(R @ K_d @ R.transpose(), decimals=2))
print("K adjoint: \n ", np.round(Ad @ K_d @ np.linalg.pinv(Ad), decimals=2))

X.plot()
X_d.plot()
plt.show()
