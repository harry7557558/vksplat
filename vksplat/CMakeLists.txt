cmake_minimum_required(VERSION 3.16)
project(VkSplat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "/W4")
    # set(CMAKE_CXX_FLAGS_DEBUG "/Zi /fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    # set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer -fno-common")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Find required packages
find_package(Vulkan REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # If pybind11 is not found, download it
    message(STATUS "Downloading pybind11")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find GLM
find_package(glm QUIET)
if(NOT glm_FOUND)
    # If GLM is not found, download it
    message(STATUS "Downloading GLM")
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG        0.9.9.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glm)
endif()

# Source files
set(SOURCES
    src/buffer.cpp
    src/colmap_reader.cpp
    src/knn.cpp
    src/scheduler.cpp
    src/gs_pipeline.cpp
    src/gs_renderer.cpp
    src/gs_trainer.cpp
    src/perf_timer.cpp
    src/python_bindings.cpp
)

set(HEADERS
    src/config.h
    src/buffer.h
    src/colmap_reader.h
    src/knn.h
    src/scheduler.h
    src/gs_pipeline.h
    src/gs_renderer.h
    src/gs_trainer.h
    src/perf_timer.h
)

# Create the Python module
pybind11_add_module(vksplat ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(vksplat PRIVATE 
    Vulkan::Vulkan
    glm::glm
    # asan
)

# Compiler-specific options
target_compile_definitions(vksplat PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

if(DEFINED OPTIMIZATION_LEVEL)
    target_compile_definitions(vksplat PRIVATE
        "OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL}")
endif()

# Set properties
set_target_properties(vksplat PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(vksplat PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(vksplat PRIVATE VK_USE_PLATFORM_XLIB_KHR)
elseif(APPLE)
    target_compile_definitions(vksplat PRIVATE VK_USE_PLATFORM_MACOS_MVK)
endif()

# Install target
install(TARGETS vksplat DESTINATION .)
