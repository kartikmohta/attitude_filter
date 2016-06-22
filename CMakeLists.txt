cmake_minimum_required(VERSION 2.8.3)
project(attitude_filter)

# Enable c++11 flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_package(catkin_simple REQUIRED)
catkin_simple()

find_package(Eigen REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

cs_add_library(attitude_cdkf src/attitude_cdkf.cpp)

cs_add_library(attitude_cdkf_nodelet src/attitude_cdkf_nodelet.cpp)
target_link_libraries(attitude_cdkf_nodelet attitude_cdkf)
