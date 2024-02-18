import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
from helpers.ec_reparametrization import ec_reparametrization

phi_vec = np.arange(-3.1 * np.pi, 3.1 * np.pi, 1e-3)

X = np.zeros((phi_vec.size, 3))
X_rp = np.zeros((phi_vec.size, 3))
for i, phi in enumerate(phi_vec):
    X[i, :] = sm.SO3.Rz(phi).log(twist="true")
    X_rp[i, :] = ec_reparametrization(X[i, :])


_, axs = plt.subplots(1, 2)
axs[0].plot(phi_vec, X, label=["ec_x", "ec_y", "ec_z"])
axs[0].legend(loc="upper right")
axs[0].set_title("Exponential Coordinates for a z-Rotation by phi")
axs[0].set(xlabel="phi[rad]")
axs[1].plot(phi_vec, X_rp, label=["ec_x", "ec_y", "ec_z"])
axs[1].legend(loc="upper right")
axs[1].set_title("Reparametrized Exponential Coordinates for a z-Rotation by phi")
axs[1].set(xlabel="phi[rad]")
plt.show()
