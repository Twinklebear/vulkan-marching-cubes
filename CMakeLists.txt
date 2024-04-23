cmake_minimum_required(VERSION 3.21)
project(vulkan_marching_cubes)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(cmake/glm.cmake)

find_package(Vulkan REQUIRED)

add_subdirectory(src)

