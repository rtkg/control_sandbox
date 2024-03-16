import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt

X = sm.SE3()
X_d = sm.SE3.Rz(-np.pi / 2, t=np.array([0.5, 0.5, 0.0]))
X_e = X.inv() * X_d  # error pose
K_d = np.diag([10, 10, 10, 1, 1, 1])
Ad = X_e.Ad()

# pfaffian constraint matrix leaving only v_x unconstrained
A = np.eye(6)
A = np.delete(A, 5, 0)

print("Adjoint: \n", np.round(Ad, decimals=2))

print("A: \n", A)
print("A @ Ad: \n", np.round(A @ Ad, decimals=2))

R = Ad
R[0:3, 3:6] = np.zeros((3, 3))
print("A @ R: \n", np.round(A @ R, decimals=2))

K_d = np.diag([1, 2, 3, 4, 5, 6])
print("K_d: \n", K_d)
print("K: \n ", np.round(R @ K_d @ R.transpose(), decimals=2))
print("K adjoint: \n ", np.round(Ad @ K_d @ Ad.transpose(), decimals=2))

X.plot()
X_d.plot()
plt.show()
