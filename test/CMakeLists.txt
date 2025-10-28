cmake_minimum_required(VERSION 3.24)
project(tiny_nn_tests LANGUAGES CXX CUDA)

# ===================== TESTS =====================
# Enable CTest
enable_testing()

file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/test/sources/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/test/headers/*.hpp")

# Add test executable
add_executable(unit_tests ${SOURCES} ${HEADERS})

# Visual Studio filters
source_group("Source Files" FILES ${SOURCES})
source_group("Header Files" FILES ${HEADERS})

# Include headers from main project
target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/code/headers
    ${CMAKE_SOURCE_DIR}/external
)

# Require C++20 on target
target_compile_features(unit_tests PUBLIC cxx_std_20)

# Link the main library
target_link_libraries(unit_tests PRIVATE tiny_nn_lib)

# Set default startup project for Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT unit_tests)

# Register the test with CTest
add_test(
    NAME unit_tests
    COMMAND unit_tests
)
