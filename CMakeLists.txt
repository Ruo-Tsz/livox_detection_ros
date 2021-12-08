cmake_minimum_required(VERSION 2.8.3)
project(livox_detection)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_MATH_DISABLE_FLOAT128")

find_package(catkin REQUIRED COMPONENTS
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    itri_msgs
    tf
    jsk_recognition_msgs
    jsk_rviz_plugins
)

find_package(PCL  REQUIRED)

catkin_package(CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
    pcl_ros
    itri_msgs
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

link_directories(
    ${PCL_LIBRARY_DIRS}
)

add_subdirectory(livox_detection/utils/lib_cpp)
