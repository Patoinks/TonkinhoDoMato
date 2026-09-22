cmake_minimum_required(VERSION 3.18)
project(tiny_opponent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate pybind11 through the installed Python package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tiny_opponent engine.cpp bindings.cpp)

if(MSVC)
  target_compile_options(tiny_opponent PRIVATE /O2 /EHsc)
else()
  target_compile_options(tiny_opponent PRIVATE -O3)
endif()

# Drop the .pyd/.so next to the sources so `import tiny_opponent` works from
# the project root without an install step.
set_target_properties(tiny_opponent PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/.."
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/..")
