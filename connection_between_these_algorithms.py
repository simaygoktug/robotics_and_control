## 1. Differential Drive Odometry

**Teori:** Diferansiyel tahrik sistemlerinde, iki tekerleğin açısal hızları kullanılarak robotun pozisyon ve yönelimi hesaplanır.

**Matematiksel Model:**
```
v = (v_right + v_left) / 2  (doğrusal hız)
ω = (v_right - v_left) / L  (açısal hız)
```

**Sorunları:** Kayma, tekerlek yarıçapı hataları, enkoder gürültüsü nedeniyle hata birikimi.

## 2. Kalman Filtering for Sensor Fusion

**Teori:** Gauss dağılımlarını kullanarak belirsizlik altında optimal durum tahmini yapar.

**İki Aşama:**
- **Prediction:** Önceki durumdan hareketle tahmin
- **Update:** Sensör ölçümü ile düzeltme

**Gaussian Belief:** μ (ortalama) ve σ² (varyans) ile temsil edilir.

## 3. Sensor Fusion: IMU + Odometry + Encoder

**Teorik Yaklaşım:**
- **IMU:** Açısal hız ve ivme bilgisi (yüksek frekanslı, kısa vadeli doğru)
- **Odometry:** Pozisyon tahmini (uzun vadede drift)
- **Encoder:** Tekerlek dönüşü (doğrudan ölçüm)

**Fusion Stratejisi:** Kalman filtresi her sensörün güvenilirlik derecesine göre ağırlıklı ortalama alır.

## 4. SLAM (Simultaneous Localization and Mapping)

**Temel Problem:** Robot hem çevreyi haritalıyor hem de bu haritada kendini konumlandırıyor.

**Matematiksel Formülasyon:**
- **Posterior:** P(x₁:ₜ, m | z₁:ₜ, u₁:ₜ)
- x: robot pozisyonları, m: harita, z: ölçümler, u: kontrol komutları

**İki Ana Yaklaşım:**
- **Online SLAM:** Sadece güncel pozisyonu tahmin et
- **Full SLAM:** Tüm geçmişi optimize et

## 5. Path Planning Algoritmaları

### A* (A-star)
**Teori:** Best-first search + heuristic
```
f(n) = g(n) + h(n)
```
- g(n): başlangıçtan n'e maliyet
- h(n): n'den hedefe heuristik maliyet

### Dijkstra
**Teori:** Uniform cost search, her yöne eşit genişleme
- A*'dan farkı: h(n) = 0

### RRT* (Rapidly-exploring Random Tree*)
**Teori:** Rastgele örnekleme + yeniden kablolama
- **RRT'den farkı:** Optimal yolları bulma garantisi

## 6. Localization: MCL vs AMCL

### Monte Carlo Localization (MCL)
**Teori:** Particle filter kullanarak belief distribution temsili
- Her particle bir pozisyon hipotezi
- Ölçümlerle particle'lar yeniden ağırlıklandırılır

### Adaptive MCL (AMCL)
**Teori:** Dinamik particle sayısı
- Belirsizlik azaldığında particle sayısını düşür
- Belirsizlik arttığında artır

## 7. Occupancy Grid Mapping

**Teori:** Çevreyi grid hücrelerine böl, her hücre için doluluk olasılığı hesapla.

**Log-odds Representation:**
```
l(m_i) = log(p(m_i) / (1 - p(m_i)))
```

**Bayes Update:** Yeni ölçümlerle grid güncellenir.

## 8. Trajectory Tracking

### Pure Pursuit
**Teori:** Robotun önünde sabit mesafede bir hedef nokta seç, ona doğru yönlen.
```
δ = arctan(2 * L * sin(α) / l_d)
```

### Stanley Controller
**Teori:** Cross-track error + heading error'u minimize et.
```
δ = ψ + arctan(k * e / v)
```

---

## BİRBİRLERİYLE BAĞLANTILAR VE KULLANIM

### 1. Temel Algılama Katmanı
```
Encoders → Odometry → Kalman Filter ← IMU
                           ↓
                    Fused Position
```

### 2. SLAM Entegrasyonu
```
Fused Position + Laser/Camera → SLAM → Map + Refined Position
```

### 3. Navigasyon Pipeline
```
SLAM Map → Path Planning → Trajectory → Trajectory Tracking
    ↑                                         ↓
Localization ← Sensor Fusion ← Control Commands
```

### 4. Gerçek Zamanlı Döngü
```
1. Sensor readings (IMU, encoders, lidar)
2. Sensor fusion (Kalman filter)
3. Localization update (AMCL)
4. SLAM update (map + position refinement)
5. Path planning (if new goal)
6. Trajectory tracking (Pure Pursuit/Stanley)
7. Control commands → actuators
```

## TEKNİK FARKLAR VE SEÇIM KRİTERLERİ

### **Deterministik vs Probabilistik**
- **Deterministik:** Odometry, Pure Pursuit
- **Probabilistik:** Kalman Filter, SLAM, MCL

### **Global vs Local**
- **Global:** A*, Dijkstra (tüm harita)
- **Local:** RRT*, Stanley (yerel planlama)

### **Online vs Offline**
- **Online:** AMCL, Trajectory tracking
- **Offline:** Grid mapping, Path planning

### **Computational Complexity**
- **Düşük:** Odometry, Pure Pursuit
- **Orta:** Kalman Filter, Stanley
- **Yüksek:** SLAM, RRT*

Bu sistemler genellikle **layered architecture** şeklinde birlikte kullanılır: alt seviyede sensor fusion, orta seviyede SLAM ve localization, üst seviyede path planning ve control.