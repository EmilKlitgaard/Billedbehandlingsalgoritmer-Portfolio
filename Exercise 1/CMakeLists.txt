cmake_minimum_required(VERSION 3.5)
project(Exercise_1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executablepickupTest
add_executable(Executable main.cpp)

# Link against the installed UR RTDE library
target_link_libraries(Executable ${OpenCV_LIBS})