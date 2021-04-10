cmake_minimum_required(VERSION 2.6)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

project(icp)
find_package(Ceres  REQUIRED)
find_package(Eigen3  REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
add_executable(icp main.cpp)
target_link_libraries(icp ${CERES_LIBRARIES} ${EIGEN3_LIBRARIES})

install(TARGETS icp RUNTIME DESTINATION bin)
