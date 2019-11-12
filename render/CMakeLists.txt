cmake_minimum_required(VERSION 2.8.6)
project(render)

link_directories(../core/)
include_directories(../core/)

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(PythonLibs REQUIRED)
find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")
add_executable(render ${srcs})
target_link_libraries(render ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut mss)
