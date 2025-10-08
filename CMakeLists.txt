cmake_minimum_required(VERSION 3.24)
project(tiny-nn LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect CUDA Toolkit automatically if not set by user
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    if(DEFINED ENV{CUDA_PATH})
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH} CACHE PATH "CUDA Toolkit Root Dir")
    else()
        message(STATUS "CUDA_TOOLKIT_ROOT_DIR not defined, relying on CMake automatic detection")
    endif()
else()
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "CUDA Toolkit Root Dir")
endif()

# Allow user to set CUDA architectures from the command line, default to AUTO
set(CMAKE_CUDA_ARCHITECTURES native)

# Try enabling C++20 for CUDA if supported by your toolkit
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

# Runtime libraries: prefer shared CUDA runtime to avoid CRT conflicts
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if (MSVC)
  # Ensure dynamic MSVC runtime (/MD for Release, /MDd for Debug)
  # Use generator expression so Debug -> MultiThreadedDebugDLL, others -> MultiThreadedDLL
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Warnings (C++ only; avoid passing MSVC flags directly to NVCC)
if (MSVC)
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
	add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/external "${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

# Source files
file(GLOB SRC_CPP "${CMAKE_SOURCE_DIR}/code/sources/*.cpp")
file(GLOB SRC_CU  "${CMAKE_SOURCE_DIR}/code/sources/*.cu")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/code/headers/*.hpp")

# Create executable
add_executable(tiny-nn ${SRC_CPP} ${SRC_CU} ${HEADERS})

# Visual Studio filters
source_group("Source Files" FILES ${SRC_CPP})
source_group("CUDA Files" FILES ${SRC_CU})
source_group("Header Files" FILES ${HEADERS})

# Set default startup project for Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT tiny-nn)

# Ensure separate compilation for CUDA
set_target_properties(tiny-nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Require C++20 on target
target_compile_features(tiny-nn PUBLIC cxx_std_20)

# Define DEBUG only in Debug builds (cross-platform)
target_compile_definitions(tiny-nn PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
	PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)