cmake_minimum_required(VERSION 3.12)
project(async_operations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(async_operations async_operations.cpp)
target_include_directories(async_operations PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(async_operations PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

find_package(Threads REQUIRED)
