cmake_minimum_required(VERSION 2.8.3)
project(livox_object_filter)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    itri_msgs
    tf
    jsk_recognition_msgs
#    rosinterface
)

find_package(OpenCV REQUIRED)

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
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    ${PCL_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
)

add_library(${PROJECT_NAME}
    src/object_filter.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)

add_executable(${PROJECT_NAME}_node
    src/main.cpp
)

target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
)
