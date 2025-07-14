#Simultaneous Localization and Mapping

#Robot hem harita çıkarır hem de kendi pozisyonunu bulur.
#Giriş: Lidar, Odometry
#Çıkış: Harita + pozisyon

#ROS Noetic ile Gmapping:

roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping

#Haritayı Kaydet

rosrun map_server map_saver -f ~/map
