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
pybind11_add_module(tonkai_fast ../tonkai_fast/fast.cpp)
set_target_properties(tonkai_fast PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(MSVC)
  target_compile_options(tiny_opponent PRIVATE /EHsc $<$<CONFIG:Release>:/O2>)
  target_compile_options(tonkai_fast PRIVATE /EHsc /fp:strict $<$<CONFIG:Release>:/O2>)
else()
  target_compile_options(tiny_opponent PRIVATE $<$<CONFIG:Release>:-O3>)
  target_compile_options(tonkai_fast PRIVATE $<$<CONFIG:Release>:-O3> -ffp-contract=off -fno-fast-math)
endif()

# Drop the .pyd/.so next to the sources so `import tiny_opponent` works from
# the project root without an install step.
set_target_properties(tiny_opponent tonkai_fast PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/.."
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/..")
