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

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/bin/shaders)

file(GLOB_RECURSE GLSL_SHADERS
    CONFIGURE_DEPENDS
    ${SHADER_SRC_DIR}/*.glsl
)

function(get_shader_stage INPUT OUTPUT)
    if (${INPUT} MATCHES "_vert\\.glsl$")
        set(${OUTPUT} vertex PARENT_SCOPE)
    elseif (${INPUT} MATCHES "_frag\\.glsl$")
        set(${OUTPUT} fragment PARENT_SCOPE)
    elseif (${INPUT} MATCHES "_comp(ute)?\\.glsl$")
        set(${OUTPUT} compute PARENT_SCOPE)
    elseif (${INPUT} MATCHES "/compute/")
        set(${OUTPUT} compute PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Cannot infer shader stage for ${INPUT}")
    endif()
endfunction()

set(SPIRV_SHADERS)

foreach(GLSL ${GLSL_SHADERS})
    get_shader_stage(${GLSL} SHADER_STAGE)

    file(RELATIVE_PATH REL_PATH ${SHADER_SRC_DIR} ${GLSL})
    get_filename_component(FILE_NAME_WE ${REL_PATH} NAME_WE)
    get_filename_component(DIR_NAME     ${REL_PATH} DIRECTORY)

    set(OUT_DIR ${SHADER_BIN_DIR}/${DIR_NAME})
    set(SPIRV   ${OUT_DIR}/${FILE_NAME_WE}.spv)

    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
        COMMAND glslc
            -fshader-stage=${SHADER_STAGE}
            --target-env=vulkan1.1
            ${GLSL}
            -o ${SPIRV}
        DEPENDS ${GLSL}
        COMMENT "Compiling shader ${REL_PATH}"
        VERBATIM
    )

    list(APPEND SPIRV_SHADERS ${SPIRV})
endforeach()

add_custom_target(shaders ALL
    DEPENDS ${SPIRV_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E rm -rf $ENV{HOME}/.local/state/corender/shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{HOME}/.local/state/corender
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_BIN_DIR}
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
