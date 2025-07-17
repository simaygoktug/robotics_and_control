#MCL, rastgele parçacıklarla robotun konum olasılığını hesaplar. ROS'ta amcl kullanılır.

roslaunch turtlebot3_navigation turtlebot3_navigation.launch

#Her parçacık bir olası robot pozisyonudur.

#Her sensör okuması ile ağırlıklar güncellenir.

#Zayıf parçacıklar silinir, güçlüler çoğaltılır.