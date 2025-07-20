#Lateral error + heading error’u birlikte kullanır. Özellikle otonom araçlarda popüler.

#Stanley algoritması, araç dinamiklerini temel alarak yol üzerindeki yönelme farkını (heading error) ve lateral hatayı (çapraz sapma) düzeltir.

#δ: direksiyon açısı (kontrol çıkışı)
#Qe: yönelme hatası (hedef yön - araç yönü)
#e: lateral hata (yol üzerindeki en yakın noktaya olan dik mesafe)
#k: kontrol kazancı (hassasiyet)
#v: hız
#epsilon: sıfıra bölmeyi engelleyen küçük sayı

import numpy as np
import matplotlib.pyplot as plt

# Sabitler
k = 1.0           # Stanley kazancı
dt = 0.1          # zaman adımı
L = 2.5           # aks mesafesi
v = 10.0          # sabit hız [m/s]
epsilon = 1e-5    # sıfırdan kaçınmak için

# Başlangıç durumu
x, y, yaw = 0.0, -3.0, np.radians(0)  # araç başta yol dışında
history_x, history_y = [x], [y]

# Takip edilecek yol (düz bir çizgi: y = 0)
path_x = np.linspace(0, 50, 500)
path_y = np.zeros_like(path_x)
path_yaw = np.zeros_like(path_x)

# Stanley Kontrol Fonksiyonu
def stanley_control(x, y, yaw, path_x, path_y, path_yaw):
    # En yakın nokta
    dx = path_x - x
    dy = path_y - y
    dists = np.hypot(dx, dy)
    idx = np.argmin(dists)

    # Heading error
    theta_e = path_yaw[idx] - yaw
    theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))  # normalize et

    # Cross-track error
    e = dists[idx]
    if (dy[idx] * np.cos(yaw) - dx[idx] * np.sin(yaw)) < 0:
        e *= -1

    # Direksiyon açısı
    delta = theta_e + np.arctan2(k * e, v + epsilon)
    return delta

# Simülasyon
for _ in range(500):
    delta = stanley_control(x, y, yaw, path_x, path_y, path_yaw)

    # Araç modeli (kendi basit bisiklet modeli)
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += (v / L) * np.tan(delta) * dt

    history_x.append(x)
    history_y.append(y)

# Görselleştirme
plt.plot(path_x, path_y, 'k--', label="Hedef Yol")
plt.plot(history_x, history_y, 'b', label="Araç İzlediği Yol")
plt.title("Stanley Controller ile Takip")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid()
plt.legend()
plt.show()

#########################################################################################

#Araç yoldan 3 metre dışta başlıyor, ama Stanley algoritması onu zamanla merkeze çekiyor.
#Cross-track error ve heading error birleşimi ile düzgün ve yumuşak bir takip yapılıyor.
#k kontrol kazancı artırılırsa daha agresif düzeltmeler olur.

#Gerçek bir GPS/Lidar destekli otonom robot üzerinde uygulamak için ROS 2’de ackermann_msgs ile bağlantı yapılabilir.
#Gerçek zamanlı olarak odometry, tf, ve map verileriyle birlikte StanleyControllerNode yazılabilir.
#A* veya RRT ile bulunan yolu bu kontrolcüyle takip ettirerek bir tam "Motion Planning + Tracking" sistemi kurulabilir.