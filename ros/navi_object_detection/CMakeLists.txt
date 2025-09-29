cmake_minimum_required(VERSION 3.16...3.26)
project(navi_object_detection VERSION 1.0.4 LANGUAGES CXX)

set(ignore ${CATKIN_INSTALL_INTO_PREFIX_ROOT})
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Patchwork++ core (그대로)
set(PARENT_PROJECT_NAME navi_object_detection)
set(TARGET_NAME ground_seg_cores)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../cpp ${CMAKE_CURRENT_BINARY_DIR}/patchworkpp_cpp)
else()
  cmake_minimum_required(VERSION 3.18)
  message(STATUS "Performing out-of-tree build, fetching Patchwork++ v${CMAKE_PROJECT_VERSION} Release from Github")
  include(FetchContent)
endif()

find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rcutils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)  # server 쪽에서 사용
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# --- OBB/클러스터링 노드 의존성 ---
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation features kdtree)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# -------------------------------
# OBB 실행 노드 (standalone)
# -------------------------------
# ※ object_detector.cpp 안에 rclcpp_components 매크로/헤더 포함이 남아있다면 지워주세요.
#    (예: rclcpp_components/register_node_macro.hpp, RCLCPP_COMPONENTS_REGISTER_NODE(...))

add_executable(obb_extractor_node
  src/obb_extractor_main.cpp
  src/object_detector.cpp
  src/lshaped_fitting.cpp
)
target_compile_features(obb_extractor_node PUBLIC cxx_std_20)
target_include_directories(obb_extractor_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(obb_extractor_node
  "rclcpp"
  "sensor_msgs"
  "visualization_msgs"
  "pcl_conversions"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
)
target_link_libraries(obb_extractor_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS} # OpenCV 라이브러리 링크 추가
)

install(TARGETS obb_extractor_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 파이썬 스크립트(있다면)
install(PROGRAMS
  scripts/dbscan_obb.py
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
# 헤더 설치(선택)
install(DIRECTORY include/ DESTINATION include/)

ament_package()
