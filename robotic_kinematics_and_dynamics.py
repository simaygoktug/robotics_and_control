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

#Düz Kinematik (Forward Kinematics – FK) – DH Parametreleri - 2 DOF Planar Robot Kolu (θ1, θ2)
#T-1=Rotz(Q1)*Transz(d)*Transx(ai)*Rotx(alfa1)

import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics(theta1, theta2, L1=1.0, L2=1.0):
    # Radyan cinsinden açıları hesapla
    theta1 = np.radians(theta1)
    theta2 = np.radians(theta2)

    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    return (0, 0), (x1, y1), (x2, y2)

# Örnek açı seti
theta1 = 45
theta2 = 30
joint1, joint2, end_effector = forward_kinematics(theta1, theta2)

# Görselleştirme
x_vals = [joint1[0], joint2[0], end_effector[0]]
y_vals = [joint1[1], joint2[1], end_effector[1]]

plt.plot(x_vals, y_vals, '-o')
plt.title("2 DOF Robot Kol – Düz Kinematik")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.axis("equal")
plt.show()

###################################################################

#Ters Kinematik (Inverse Kinematics – IK) - Amaç: Uç nokta (x, y) verildiğinde Q1 ve Q2 açılarını bulmak.
#Pozisyon Çözümleme

def inverse_kinematics(x, y, L1=1.0, L2=1.0):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1:
        raise ValueError("Erişilemeyen pozisyon")

    theta2 = np.arccos(D)
    theta1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(theta2), L1+L2*np.cos(theta2))

    return np.degrees(theta1), np.degrees(theta2)

# Test: Uç efektör 1.5, 0.5 koordinatına ulaşsın
theta1, theta2 = inverse_kinematics(1.5, 0.5)
print(f"Ters Kinematik Açıları: θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}°")

# Ulaşımı görselleştirelim
_, _, eff = forward_kinematics(theta1, theta2)
print(f"Ulaşılan nokta: {eff}")

###################################################################

#Dinamik Modellemler (Lagrange Yöntemiyle)
#L=T-V (Lagrangian = Kinetik - Potansiyel Enerji)
#2 DOF Planar Robot Kolun Lagrangian'ı
#Bu formdan Euler-Lagrange denklemine geçip diferansiyel denklemler türetilebilir. Ancak kontrol için bu form genellikle M(q)q_dd + C(q,q_d)q_d + G(q) = τ yapısına çevrilir.

import sympy as sp

# Değişkenler
θ1, θ2 = sp.symbols('θ1 θ2', real=True)
θ1d, θ2d = sp.symbols('θ1d θ2d', real=True)
L1, L2, m1, m2, g = sp.symbols('L1 L2 m1 m2 g', positive=True)

# Kinetik ve potansiyel enerji
T1 = (1/2)*m1*(L1**2)*(θ1d**2)
T2 = (1/2)*m2*((L1**2)*(θ1d**2) + (L2**2)*(θ2d**2) + 2*L1*L2*θ1d*θ2d*sp.cos(θ2))
T = T1 + T2

V1 = m1 * g * L1 * sp.sin(θ1)
V2 = m2 * g * (L1 * sp.sin(θ1) + L2 * sp.sin(θ1 + θ2))
V = V1 + V2

L = T - V
sp.pprint(sp.simplify(L))

#Lagrangian enerji bazlı formülasyondur. Sistem davranışını tanımlar. Bu diferansiyel denklem, genellikle nonlineer ve karmaşık formda çıkar.
#AMAÇ: Sistemin ivmesini q olarak çözmek. Sonra kontrol yasası uygulamak.
#M+C+G=τ formuna geçilir. Sayısal integratörlerle çözülmeye uygun (örneğin Runge-Kutta), Kontrol algoritmaları ile doğrudan uyumlu (örneğin: Computed Torque, LQR, PID), Matlab, ROS, Python'daki simülasyon motorlarının beklediği form.
#Sistem lineerleştirilerek bu şekilde doğrudan transfer fonksiyonu elde edilir. Frekans cevabı, kök yer eğrisi (root locus), Bode diyagramı gibi analizler için uygun hale gelir.

###################################################################

#ROS + Gazebo Simülasyonu (URDF + RViz)
#URDF Modeli Oluştur – my_robot.urdf.xacro

<robot name="two_link_robot">
  <link name="base_link">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <link name="link1">
    <visual><geometry><cylinder radius="0.02" length="1.0"/></geometry></visual>
  </link>
  <link name="link2">
    <visual><geometry><cylinder radius="0.02" length="1.0"/></geometry></visual>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>

#RViz veya Gazebo ile Göster

roslaunch your_package display.launch
# display.launch içinde robot_state_publisher + joint_state_publisher + rviz

#Kinematik Simülasyonu 
#Python’dan rospy ile açı gönder:

import rospy
from std_msgs.msg import Float64

rospy.init_node("angle_sender")
pub1 = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=10)
pub2 = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=10)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    pub1.publish(1.0)  # 1 rad
    pub2.publish(0.5)
    rate.sleep()

#Bu kinematik hesaplayıcıyı web arayüzüne bağlanabilir (Flask + Plotly),
#ROS MoveIt ile ters kinematik çözücü kullanarak bir robot kol simülasyonu yapılabilir,
#Örnekleri Jupyter Notebook haline getirip interaktif grafikleştirilebilir.