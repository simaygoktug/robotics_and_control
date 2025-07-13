#Kalman filtreleme, gürültülü sensör verilerinden en iyi kestirimi yapar.

######################################################################################

import numpy as np
import matplotlib.pyplot as plt

n = 100
true_pos = np.cumsum(np.ones(n) * 0.5)  # sabit hızla giden robot

# Gürültülü ölçümler
imu = true_pos + np.random.normal(0, 0.2, n)
encoder = true_pos + np.random.normal(0, 0.1, n)

# Kalman Değişkenleri
x_est = 0
P = 1
R = 0.15  # ölçüm gürültüsü
Q = 0.01  # model gürültüsü
fused = []

for i in range(n):
    # Tahmin
    x_pred = x_est
    P_pred = P + Q

    # Ölçüm
    z = (imu[i] + encoder[i]) / 2

    # Güncelleme
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred
    fused.append(x_est)

plt.plot(true_pos, label="Gerçek")
plt.plot(imu, label="IMU")
plt.plot(encoder, label="Encoder")
plt.plot(fused, label="Kalman")
plt.legend()
plt.grid()
plt.title("Kalman Filter ile Sensor Fusion")
plt.show()

######################################################################################

#IMU: ivmeölçer + jiroskop
#Odometry: tekerlek pozisyonu
#Encoder: tekerlek dönüş bilgisi

#Bu sensörleri yukarıdaki Kalman yapısı ile birleştirebiliriz. Gerçek sistemde ROS topic'lerinden veri toplanır.
rostopic echo /imu/data
rostopic echo /odom
rostopic echo /joint_states

#Bu verilerle bir Extended Kalman Filter (EKF) uygulanır. ROS’ta robot_localization paketi bu işi yapar.
<node pkg="robot_localization" type="ekf_localization_node" name="ekf">
  <param name="use_imu" value="true"/>
  <param name="use_odometry" value="true"/>
</node>

