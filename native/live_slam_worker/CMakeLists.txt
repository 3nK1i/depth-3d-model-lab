cmake_minimum_required(VERSION 3.16)
project(eia_stella_live_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(stella_vslam CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(eia_stella_live_worker main.cc)

if(TARGET stella_vslam::stella_vslam)
    set(EIA_STELLA_TARGET stella_vslam::stella_vslam)
elseif(TARGET stella_vslam)
    set(EIA_STELLA_TARGET stella_vslam)
else()
    message(FATAL_ERROR "Could not resolve exported stella_vslam CMake target")
endif()

set(EIA_STELLA_VERSION "0.3.13" CACHE STRING "Qualified stella_vslam version")
set(EIA_STELLA_COMMIT "unknown" CACHE STRING "Qualified stella_vslam commit")

target_compile_definitions(eia_stella_live_worker PRIVATE
    EIA_STELLA_VERSION="${EIA_STELLA_VERSION}"
    EIA_STELLA_COMMIT="${EIA_STELLA_COMMIT}"
)

target_compile_options(eia_stella_live_worker PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(eia_stella_live_worker PRIVATE
    ${EIA_STELLA_TARGET}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
if(UNIX AND NOT APPLE)
    target_link_libraries(eia_stella_live_worker PRIVATE rt)
endif()

install(TARGETS eia_stella_live_worker RUNTIME DESTINATION bin)
