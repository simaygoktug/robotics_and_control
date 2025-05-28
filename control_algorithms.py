#PID Kontrol (Zaten ustasın)

import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

G = ctrl.TransferFunction([1], [1, 10, 20])  # Örnek sistem
Kp, Ki, Kd = 1.0, 0.5, 0.1
PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
system = ctrl.feedback(PID * G, 1)

t, y = ctrl.step_response(system)
plt.plot(t, y)
plt.title("PID Kontrollü Sistem")
plt.grid()
plt.show()

#İleri Seviye Kontrol Algoritmaları

#State Space Representation (Durum Uzayı Modeli) x(t): sistemin iç durumu (state vector), u(t): giriş (input), y(t): çıkış (output)

#Model Predictive Control (MPC) MPC, gelecek adımları tahmin ederek optimize eder.
#Prediction horizonda toplam sembolü((y-r)^2+(lamda(kontrol sinyali maliyet katsayısı)*deltau)^2)

#Adaptive Control (MRAC, Gain Scheduling) Amaç, kontrol edilen sistemin davranışını bir referans modele yaklaştırmak. 
#Q=Adaptif Kazaç=-gama*x*(y-yreferans)

#LQR (Linear Quadratic Regulator) LQR, maliyet fonksiyonunu minimize eden geri besleme kazancı K'yi bulur

###################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Sistem parametreleri
A = np.array([[-2]])
B = np.array([[1]])
C = np.array([[1]])
D = np.array([[0]])

# Durum uzayı modeli oluştur
sys = signal.StateSpace(A, B, C, D)

# Step response
t, y = signal.step(sys)

# Görselleştirme
plt.plot(t, y)
plt.title("State Space Step Response")
plt.xlabel("Zaman (s)")
plt.ylabel("Çıkış y(t)")
plt.grid()
plt.show()

###################################################################

import numpy as np
import matplotlib.pyplot as plt

# Model parametreleri
a = 0.9  # çıkış geri beslemesi
b = 0.1  # giriş etkisi
y = [0]
u = [0]

setpoint = 1.0
Np = 10
λ = 0.1

for k in range(30):
    error = setpoint - y[-1]
    # MPC kontrol: sadece en yakın adımı kontrol et
    u_next = b * error / (b**2 + λ)
    u.append(u_next)
    y_next = a * y[-1] + b * u_next
    y.append(y_next)

plt.plot(y, label="Çıkış")
plt.plot([setpoint]*len(y), 'r--', label="Referans")
plt.title("Basit MPC Simülasyonu")
plt.xlabel("Zaman Adımı")
plt.grid()
plt.legend()
plt.show()

###################################################################

import numpy as np
import matplotlib.pyplot as plt

a = 1.0  # gerçek sistem
am = 2.0  # referans model
gamma = 5.0  # öğrenme oranı

x = 0
xm = 0
theta = 0
u_hist, x_hist, xm_hist = [], [], []

for _ in range(100):
    r = 1  # referans
    u = theta * r
    x = x + 0.01 * (-a * x + u)
    xm = xm + 0.01 * (-am * xm + r)
    
    e = x - xm
    theta = theta - 0.01 * gamma * r * e
    
    x_hist.append(x)
    xm_hist.append(xm)
    u_hist.append(u)

plt.plot(x_hist, label="Gerçek Sistem")
plt.plot(xm_hist, label="Referans Model")
plt.title("MRAC ile Adaptif Kontrol")
plt.xlabel("Zaman")
plt.grid()
plt.legend()
plt.show()

###################################################################

import numpy as np
import control
import matplotlib.pyplot as plt

A = np.array([[0, 1],
              [0, -1]])
B = np.array([[0],
              [1]])

Q = np.diag([10, 1])  # Durum maliyeti
R = np.array([[1]])   # Giriş maliyeti

# LQR kazancı hesapla
K, _, _ = control.lqr(A, B, Q, R)

A_cl = A - B @ K  # Kapalı çevrim sistemi

sys = control.StateSpace(A_cl, B, np.eye(2), np.zeros((2,1)))
t, y = control.step_response(sys)

plt.plot(t, y[0], label="Pozisyon")
plt.plot(t, y[1], label="Hız")
plt.title("LQR Kontrollü Sistem")
plt.grid()
plt.legend()
plt.show()

###################################################################

#Robotik Kinematik ve Dinamikler

#Düz Kinematik (FK) – DH Parametreleri

#Ters Kinematik (IK) – Pozisyon çözümleme

#Dinamik Modellemler – Lagrange, Newton-Euler yöntemleri

###################################################################