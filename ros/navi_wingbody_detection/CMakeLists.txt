cmake_minimum_required(VERSION 3.16...3.26)
project(navi_wingbody_detection VERSION 1.0.4 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)

# Python & pybind11 (먼저)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

# ROS2 / 기타
find_package(nav_msgs REQUIRED)
find_package(rcutils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation features kdtree)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# ---------------------
# lshaped_core (공통 코어)
# ---------------------
add_library(lshaped_core STATIC
  src/lshaped_fitting.cpp
)
target_include_directories(lshaped_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(lshaped_core
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

# ---------------------
# obb_extractor_node
# ---------------------
add_executable(obb_extractor_node
  src/obb_extractor_main.cpp
  src/object_detector.cpp
  # src/lshaped_fitting.cpp  <-- 코어로 빠졌으니 제거
)
target_compile_features(obb_extractor_node PUBLIC cxx_std_20)
target_include_directories(obb_extractor_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(obb_extractor_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(obb_extractor_node
  lshaped_core            # 🔗 코어 링크
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

install(TARGETS obb_extractor_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# ---------------------
# object_tracker_node
# ---------------------
add_executable(object_tracker_node
  src/obb_tracker_main.cpp
  src/object_tracker.cpp
  src/sort_tracker.cpp
)
target_compile_features(object_tracker_node PUBLIC cxx_std_17)
target_include_directories(object_tracker_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(object_tracker_node
  rclcpp
  visualization_msgs
  tf2
  tf2_geometry_msgs
)
target_link_libraries(object_tracker_node
  ${OpenCV_LIBS}
)

install(TARGETS object_tracker_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# ---------------------
# pybind11 바인딩 (lshape_bindings)
# ---------------------
# ⚠️ INSTALL_DIR 인자는 사용하지 않습니다.
python3_add_library(lshape_bindings MODULE
  src/bindings/lshape_bindings.cpp
  WITH_SOABI
)
target_include_directories(lshape_bindings PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(lshape_bindings
  PRIVATE
    pybind11::module
    lshaped_core         # 🔗 코어만 링크 (OpenCV/Eigen은 코어에 이미 연결됨)
)

# 설치 경로는 install()에서 지정
install(TARGETS lshape_bindings
  LIBRARY DESTINATION lib/${PROJECT_NAME}
)


add_executable(pc_tf_repub_node
  src/pc_tf_repub_node.cpp
)
target_compile_features(pc_tf_repub_node PUBLIC cxx_std_17)
ament_target_dependencies(pc_tf_repub_node
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_link_libraries(pc_tf_repub_node
  ${PCL_LIBRARIES}
)
install(TARGETS pc_tf_repub_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)



# ---------------------
# Python 스크립트
# ---------------------
install(PROGRAMS
  scripts/imu_viz_node.py
  scripts/obb_angle_error_node.py
  scripts/object_detector.py
  scripts/pc_tf_repub.py  
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------
# 기타 리소스
# ---------------------
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)

ament_package()
