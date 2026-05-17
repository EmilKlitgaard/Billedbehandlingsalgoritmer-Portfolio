cmake_minimum_required(VERSION 3.5)
project(Exercise_2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable and add it to the project file at ../
add_executable(Executable main.cpp)

# Output executable to parent directory
set_target_properties(Executable PROPERTIES RUNTIME_OUTPUT_DIRECTORY "..")

# Link against OpenCV
target_link_libraries(Executable ${OpenCV_LIBS})