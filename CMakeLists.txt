cmake_minimum_required(VERSION 3.20)

project(corender LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(Vulkan REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

file(GLOB CORENDER_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.c)

add_library(corender STATIC
    ${CORENDER_SRCS}
    ${CMAKE_SOURCE_DIR}/vendor/vma/vma_impl.cpp
)

target_include_directories(corender
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(corender PRIVATE -g -O0 -DDEBUG)
else()
    target_compile_options(corender PRIVATE
        -O3 -DNDEBUG
        -march=native
        -flto
        -fno-plt
        -fomit-frame-pointer
        -fstrict-aliasing
    )
    target_link_options(corender PRIVATE -flto)
endif()

add_custom_target(shaders ALL
    COMMAND ${CMAKE_COMMAND} -E rm -rf $ENV{HOME}/.local/state/corender/shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/default
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/instanced
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/compute

    COMMAND glslc -fshader-stage=vertex
        ${CMAKE_SOURCE_DIR}/shaders/instanced/basic_vert.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/instanced/basic_vert.spv

    COMMAND glslc -fshader-stage=fragment
        ${CMAKE_SOURCE_DIR}/shaders/instanced/basic_frag.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/instanced/basic_frag.spv

    COMMAND glslc -fshader-stage=vertex
        ${CMAKE_SOURCE_DIR}/shaders/default/basic_vert.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/default/basic_vert.spv

    COMMAND glslc -fshader-stage=fragment
        ${CMAKE_SOURCE_DIR}/shaders/default/basic_frag.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/default/basic_frag.spv

    COMMAND glslc -fshader-stage=compute
        ${CMAKE_SOURCE_DIR}/shaders/compute/1_bin_compute.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/compute/1_bin_compute.spv

    COMMAND glslc -fshader-stage=compute
        ${CMAKE_SOURCE_DIR}/shaders/compute/2_base_parity_compute.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/compute/2_base_parity_compute.spv

    COMMAND glslc -fshader-stage=compute --target-env=vulkan1.1 
        ${CMAKE_SOURCE_DIR}/shaders/compute/3_prefix_parity_compute.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/compute/3_prefix_parity_compute.spv

    COMMAND glslc -fshader-stage=compute
        ${CMAKE_SOURCE_DIR}/shaders/compute/4_fill_compute.glsl
        -o ${CMAKE_BINARY_DIR}/bin/shaders/compute/4_fill_compute.spv

    COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{HOME}/.local/state/corender
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/bin/shaders
        $ENV{HOME}/.local/state/corender/shaders
)

file(GLOB EXAMPLE_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/examples/*.c)

foreach(example ${EXAMPLE_SRCS})
    get_filename_component(example_name ${example} NAME_WE)
    set(target_name example_${example_name})

    add_executable(${target_name} ${example})

    target_link_libraries(${target_name}
        PRIVATE
            corender
            glfw
            OpenGL::GL
            Vulkan::Vulkan
            stdc++
            glia
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -g -O0 -DDEBUG)
    else()
        target_compile_options(${target_name} PRIVATE
            -O3 -DNDEBUG
            -march=native
            -flto
            -fno-plt
            -fomit-frame-pointer
            -fstrict-aliasing
        )
        target_link_options(${target_name} PRIVATE -flto)
    endif()
endforeach()

install(TARGETS corender ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/corender DESTINATION include)

