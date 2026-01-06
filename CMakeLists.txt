cmake_minimum_required(VERSION 3.12)
project(gpulite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create header-only interface library
add_library(gpulite INTERFACE)

# Include directories
target_include_directories(gpulite INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# System dependencies
find_package(Threads REQUIRED)
if(WIN32)
    # Windows doesn't need explicit dynamic loading libraries
    target_link_libraries(gpulite INTERFACE Threads::Threads)
else()
    target_link_libraries(gpulite INTERFACE 
        ${CMAKE_DL_LIBS}
        Threads::Threads
    )
endif()

# Build examples by default
option(BUILD_EXAMPLES "Build gpulite examples" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples/basic_vector_add)
    add_subdirectory(examples/matrix_multiply)
    add_subdirectory(examples/templated_kernels)
    add_subdirectory(examples/async_operations)
endif()

# Installation rules
install(TARGETS gpulite EXPORT gpuliteConfig)
install(FILES
    gpulite/gpulite.hpp
    DESTINATION include/gpulite
)
install(EXPORT gpuliteConfig DESTINATION lib/cmake/gpulite)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gpuliteConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gpuliteConfigVersion.cmake"
    DESTINATION lib/cmake/gpulite
)