cmake_minimum_required(VERSION 3.0.2)
project(vicon2odom)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  message_filters
  mavros
  sensor_msgs
  tf2
)
find_package(Eigen3 REQUIRED)
## 声明 catkin 包
catkin_package()

## 指定编译的目标
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)

## 添加可执行文件
add_executable(vicon2odom_node src/vicon2odom.cpp)

## 链接依赖
target_link_libraries(vicon2odom_node
  ${catkin_LIBRARIES}
)

