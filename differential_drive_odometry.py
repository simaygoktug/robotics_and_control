#Differential drive robotlar 2 tekerleğe sahiptir. Tekerlek hızları biliniyorsa, robotun pozisyonu ve yönelimi güncellenebilir.

#Formüller:

#Δx = v * cos(θ) * Δt

#Δy = v * sin(θ) * Δt

#Δθ = (v_r - v_l) / L * Δt

######################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Parametreler
wheel_radius = 0.05  # 5 cm
wheel_base = 0.3     # tekerlekler arası mesafe
dt = 0.1             # zaman adımı
x, y, theta = 0, 0, 0

# Sağ ve sol tekerlek hızları (rad/s)
v_r = [1.2]*100
v_l = [1.0]*100

xs, ys = [], []

for vr, vl in zip(v_r, v_l):
    v = wheel_radius * (vr + vl) / 2
    omega = wheel_radius * (vr - vl) / wheel_base

    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += omega * dt

    xs.append(x)
    ys.append(y)

plt.plot(xs, ys)
plt.title("Differential Drive Odometry")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid()
plt.show()
