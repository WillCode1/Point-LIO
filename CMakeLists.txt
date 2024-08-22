cmake_minimum_required(VERSION 3.5)
project(point_lio)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -Wno-unused-local-typedefs -Wno-format-truncation -Wno-unused-result -Wno-pedantic -Wno-unused-parameter -Wno-unused-variable")

# echo "/home/will/coredump/core-%e-%p-%s-%t" > /proc/sys/kernel/core_pattern
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch32)|(AARCH32)|(aarch64)|(AARCH64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 6)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    add_definitions(-DBNB_PROC_NUM=6)
    message("core for MP: 4")
  elseif(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    add_definitions(-DBNB_PROC_NUM=4)
    message("core for MP: 4")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    add_definitions(-DBNB_PROC_NUM=3)
    message("core for MP: 3")
  else()
    add_definitions(-DMP_PROC_NUM=1)
    add_definitions(-DBNB_PROC_NUM=2)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
  add_definitions(-DBNB_PROC_NUM=2)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization timer thread chrono)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

link_directories(
	include
	${PCL_LIBRARY_DIRS}
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(pointlio_mapping src/laserMapping.cpp 
                src/li_initialization.cpp src/parameters.cpp src/preprocess.cpp src/Estimator.cpp 
                src/IMU_Processing.cpp
                include/backend_optimization/global_localization/scancontext/Scancontext.cpp
                include/backend_optimization/global_localization/InitCoordinate.cpp
                include/backend_optimization/interface_ros1.cpp)
target_link_libraries(pointlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} stdc++fs ${Boost_LIBRARIES} gtsam)
target_include_directories(pointlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})



