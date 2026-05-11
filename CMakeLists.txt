cmake_minimum_required(VERSION 3.20)

project(corender LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(CORENDER_ENABLE_GPU_STATS "Enable GPU stats collection in C/C++ and GLSL shaders" OFF)

if(CORENDER_ENABLE_GPU_STATS)
    set(CORENDER_GPU_STATS_VALUE 1)
else()
    set(CORENDER_GPU_STATS_VALUE 0)
endif()

message(STATUS "CORENDER_ENABLE_GPU_STATS=${CORENDER_ENABLE_GPU_STATS}")
message(STATUS "CR_ENABLE_GPU_STATS=${CORENDER_GPU_STATS_VALUE}")

find_package(Vulkan REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

find_program(GLSLC_EXECUTABLE glslc REQUIRED)

file(GLOB CORENDER_SRCS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.c
)

add_library(corender STATIC
    ${CORENDER_SRCS}
    ${CMAKE_SOURCE_DIR}/vendor/vma/vma_impl.cpp
)

target_include_directories(corender
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(corender
    PUBLIC
        CR_ENABLE_GPU_STATS=${CORENDER_GPU_STATS_VALUE}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(corender PRIVATE
        -g
        -O0
        -DDEBUG
    )

    set(SHADER_OPT_FLAGS
        -O0
        -g
    )
else()
    target_compile_options(corender PRIVATE
        -O3
        -DNDEBUG
        -march=native
        -flto
        -fno-plt
        -fomit-frame-pointer
        -fstrict-aliasing
    )

    target_link_options(corender PRIVATE
        -flto
    )

    set(SHADER_OPT_FLAGS
        -O
    )
endif()

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_CFG_DIR ${CMAKE_SOURCE_DIR}/config)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/bin/shaders)
set(GEN_DIR ${CMAKE_BINARY_DIR}/generated_shaders)

file(MAKE_DIRECTORY ${GEN_DIR})

set(SHADER_DEFINES
    -DCR_ENABLE_GPU_STATS=${CORENDER_GPU_STATS_VALUE}
)

file(GLOB_RECURSE SHADER_SHARED_DEPS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/shared/*.glsl
)

set(SUBGROUP_SIZES 4 8 16 32 64)

set(PIPELINE_tile_events
    tile_events # input buffer
    tile_events # output buffer, in-place
    6           # binding location for input buffer
    -1          # binding location for output buffer, no output buffer
    -1          # prefixed output, same as INPUT
    0           # enable 1d prefix sum
    1           # enable 2d prefix sum
    0           # DO_XOR
    1           # PREFIX_INCLUSIVE
    1           # PREFIX_SIGNED
    n_tiles_x
    n_tiles_y
)

set(PIPELINE_LIST
    PIPELINE_tile_events
)

file(GLOB PREFIX_TEMPLATES CONFIGURE_DEPENDS
    ${SHADER_CFG_DIR}/prefix*d_step*_compute.glsl.in
)

if(NOT PREFIX_TEMPLATES)
    message(FATAL_ERROR "No prefix shader templates found in ${SHADER_CFG_DIR}")
endif()

set(GENERATED_PREFIX_SHADERS)

foreach(TEMPLATE IN LISTS PREFIX_TEMPLATES)
    get_filename_component(TEMPLATE_NAME ${TEMPLATE} NAME)

    string(REGEX MATCH "prefix[12]d" PREFIX_DIM ${TEMPLATE_NAME})
    string(REGEX MATCH "step[123]" STEP_PART ${TEMPLATE_NAME})

    foreach(PNAME IN LISTS PIPELINE_LIST)
        set(PIPELINE ${${PNAME}})

        list(GET PIPELINE 0 PIPELINE_NAME)
        list(GET PIPELINE 1 INPUT)
        list(GET PIPELINE 2 INPUT_BINDING)
        list(GET PIPELINE 3 OUTPUT_BINDING)
        list(GET PIPELINE 4 PREFIXED_OUTPUT)
        list(GET PIPELINE 5 ENABLE_1D)
        list(GET PIPELINE 6 ENABLE_2D)
        list(GET PIPELINE 7 DO_XOR_VALUE)
        list(GET PIPELINE 8 PREFIX_INCLUSIVE_VALUE)
        list(GET PIPELINE 9 PREFIX_SIGNED_VALUE)
        list(GET PIPELINE 10 DIM_X)
        list(GET PIPELINE 11 DIM_Y)

        if(PREFIX_DIM STREQUAL "prefix1d" AND NOT ENABLE_1D)
            continue()
        endif()

        if(PREFIX_DIM STREQUAL "prefix2d" AND NOT ENABLE_2D)
            continue()
        endif()

        if(PREFIXED_OUTPUT STREQUAL "-1")
            unset(PREFIX_OUTPUT)
            set(PREFIXED_OUTPUT ${INPUT})
            set(OUTPUT_BINDING ${INPUT_BINDING})
        else()
            set(PREFIX_OUTPUT ON)
        endif()

        if(DO_XOR_VALUE)
            set(DO_XOR ON)
        else()
            unset(DO_XOR)
        endif()

        if(PREFIX_INCLUSIVE_VALUE)
            set(PREFIX_INCLUSIVE ON)
        else()
            unset(PREFIX_INCLUSIVE)
        endif()

        if(PREFIX_SIGNED_VALUE)
            set(PREFIX_SIGNED ON)
        else()
            unset(PREFIX_SIGNED)
        endif()

        foreach(SG IN LISTS SUBGROUP_SIZES)
            set(SUBGROUP_SIZE ${SG})

            set(OUT_GLSL
                ${GEN_DIR}/1_${PIPELINE_NAME}_${PREFIX_DIM}_${STEP_PART}_sg${SG}_compute.glsl
            )

            configure_file(
                ${TEMPLATE}
                ${OUT_GLSL}
                @ONLY
            )

            list(APPEND GENERATED_PREFIX_SHADERS ${OUT_GLSL})
        endforeach()
    endforeach()
endforeach()

file(GLOB_RECURSE DEFAULT_SHADERS CONFIGURE_DEPENDS
    ${SHADER_SRC_DIR}/default/*.glsl
)

file(GLOB_RECURSE INSTANCED_SHADERS CONFIGURE_DEPENDS
    ${SHADER_SRC_DIR}/instanced/*.glsl
)

file(GLOB_RECURSE COMPUTE_STATIC_SHADERS CONFIGURE_DEPENDS
    ${SHADER_SRC_DIR}/compute/*.glsl
)

set(COMPUTE_SHADERS
    ${COMPUTE_STATIC_SHADERS}
    ${GENERATED_PREFIX_SHADERS}
)

function(get_shader_stage INPUT OUTPUT)
    if("${INPUT}" MATCHES "_vert\\.glsl$")
        set(${OUTPUT} vertex PARENT_SCOPE)
    elseif("${INPUT}" MATCHES "_frag\\.glsl$")
        set(${OUTPUT} fragment PARENT_SCOPE)
    elseif("${INPUT}" MATCHES "compute\\.glsl$")
        set(${OUTPUT} compute PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Cannot infer shader stage for ${INPUT}")
    endif()
endfunction()

function(compile_shader_group SHADER_LIST OUT_SUBDIR OUT_VAR)
    set(SPVS)

    foreach(GLSL IN LISTS ${SHADER_LIST})
        get_shader_stage(${GLSL} STAGE)

        get_filename_component(FILE_NAME_WE ${GLSL} NAME_WE)

        set(SPIRV
            ${SHADER_BIN_DIR}/${OUT_SUBDIR}/${FILE_NAME_WE}.spv
        )

        add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_DIR}/${OUT_SUBDIR}
            COMMAND ${GLSLC_EXECUTABLE}
                ${SHADER_OPT_FLAGS}
                ${SHADER_DEFINES}
                -fshader-stage=${STAGE}
                --target-env=vulkan1.1
                ${GLSL}
                -o ${SPIRV}
            DEPENDS
                ${GLSL}
                ${SHADER_SHARED_DEPS}
            VERBATIM
            COMMENT "Compiling shader ${GLSL} -> ${SPIRV}"
        )

        list(APPEND SPVS ${SPIRV})
    endforeach()

    set(${OUT_VAR} ${SPVS} PARENT_SCOPE)
endfunction()

compile_shader_group(DEFAULT_SHADERS default SPIRV_DEFAULT)
compile_shader_group(INSTANCED_SHADERS instanced SPIRV_INSTANCED)
compile_shader_group(COMPUTE_SHADERS compute SPIRV_COMPUTE)

set(SPIRV_SHADERS
    ${SPIRV_DEFAULT}
    ${SPIRV_INSTANCED}
    ${SPIRV_COMPUTE}
)

add_custom_target(shaders ALL
    DEPENDS ${SPIRV_SHADERS}
    COMMAND ${CMAKE_COMMAND} -E rm -rf $ENV{HOME}/.local/state/corender/shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{HOME}/.local/state/corender/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_BIN_DIR}
        $ENV{HOME}/.local/state/corender/shaders
    COMMENT "Installing compiled shaders to $ENV{HOME}/.local/state/corender/shaders"
)

add_dependencies(corender shaders)

file(GLOB EXAMPLE_SRCS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/examples/*.c
)

foreach(example ${EXAMPLE_SRCS})
    get_filename_component(example_name ${example} NAME_WE)

    set(target_name example_${example_name})

    add_executable(${target_name}
        ${example}
    )

    target_link_libraries(${target_name}
        PRIVATE
            corender
            glfw
            OpenGL::GL
            Vulkan::Vulkan
            stdc++
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE
            -g
            -O0
            -DDEBUG
        )
    else()
        target_compile_options(${target_name} PRIVATE
            -O3
            -DNDEBUG
            -march=native
            -flto
            -fno-plt
            -fomit-frame-pointer
            -fstrict-aliasing
        )

        target_link_options(${target_name} PRIVATE
            -flto
        )
    endif()
endforeach()

install(TARGETS corender
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/corender
    DESTINATION include
)
