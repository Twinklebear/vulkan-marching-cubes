set(BLOCK_SIZE 512)

add_spirv_embed_library(scan_shaders
    add_block_sums.comp
    block_prefix_sum.comp
    prefix_sum.comp
	COMPILE_OPTIONS -O --target-env=vulkan1.2
    INCLUDE_DIRECTORIES
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        BLOCK_SIZE=${BLOCK_SIZE})

add_spirv_embed_library(mc_shaders
    compute_active_voxel.comp
    compute_num_verts.comp
    compute_vertices.comp
    stream_compact.comp
	COMPILE_OPTIONS -O --target-env=vulkan1.2
    INCLUDE_DIRECTORIES
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        VOLUME_DTYPE=uint
        UINT8_VOLUME=1)

add_library(vulkan_utils util.cpp vulkan_utils.cpp)

target_link_libraries(vulkan_utils PUBLIC
    Vulkan::Vulkan
    glm)

set_target_properties(vulkan_utils PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

add_executable(vulkan_marching_cubes
    marching_cubes.cpp
    exclusive_scan.cpp
    compute_marching_cubes.cpp)

set_target_properties(vulkan_marching_cubes PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_compile_options(vulkan_marching_cubes PUBLIC
    -DBLOCK_SIZE=${BLOCK_SIZE}
    -DVOLUME_DTYPE=uint
    -DUINT8_VOLUME=1)

target_link_libraries(vulkan_marching_cubes PUBLIC
    vulkan_utils
	scan_shaders
    mc_shaders)

