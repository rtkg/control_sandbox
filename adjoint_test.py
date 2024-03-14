import spatialmath as sm
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt

X = sm.SE3()
X_d = sm.SE3.Rz(-np.pi / 4, t=np.array([0.5, 0.5, 0.0]))
X_e = X.inv() * X_d  # error pose
K_d = np.diag([10, 10, 10, 1, 1, 1])
Ad = X_e.Ad()

# pfaffian constraint matrix leaving only v_x unconstrained
A = np.eye(6)
A = A[1:6, :]

print("Adjoint: \n", np.round(Ad, decimals=2))

print("A: \n", A)
print("A @ Ad: \n", np.round(A @ Ad, decimals=2))

R = Ad
R[0:3, 3:6] = np.zeros((3, 3))
print("A @ R: \n", np.round(A @ R, decimals=2))


# X.plot()
# X_d.plot()
# plt.show()
