cmake_minimum_required(VERSION 3.8)
project(planik_dnn_inference)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Python module
ament_python_install_package(${PROJECT_NAME})

# Install Python modules and scripts
install(PROGRAMS
  scripts/model_infer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  models
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
