cmake_minimum_required(VERSION 4.0)

####################  一、build  ####################

# 1. set compiler before call project()
message(STATUS "Default C++ compiler: ${CMAKE_CXX_COMPILER}")

# 2. set cpp standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "Default C++ standard: ${CMAKE_CXX_STANDARD}")

# 3. set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug) # Options: Debug, Release, RelWithDebInfo, MinSizeRel
endif()
message(STATUS "Default cmake build type: ${CMAKE_BUILD_TYPE}")


####################  二、env  ####################

# 1. project
project(MINIAIC)

# set PIC for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)


####################  三、main  ####################

# 1. include directories
set(MINIAIC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories("${MINIAIC_SOURCE_DIR}")

# 2. subdirectory
add_subdirectory(miniaic)
