#include "compute_marching_cubes.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include "mc_shaders_embedded_spv.h"
#include "tri_table.h"

const size_t MarchingCubes::max_dispatch_size = ((2 * 65535 * 4) / 256) * 256;

MarchingCubes::MarchingCubes(std::shared_ptr<vkrt::Device> &device,
                             void *volume_data,
                             const glm::uvec3 &volume_dims,
                             const std::string &volume_type)
    : device(device),
      active_voxel_scanner(device),
      num_verts_scanner(device),
      volume_dims(volume_dims)
{
    // Note: explicitly not using 3D storage textures here to match the WebGPU version, where
    // they're not implemented fully yet
    volume_data_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    compute_active_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    // Note: not using push constants here to match the WebGPU backend, where push constants
    // aren't available
    stream_compact_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(
                0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(
                1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(
                2, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(3, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    compute_num_verts_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(2, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    compute_verts_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(2, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(3, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    // Allocate the descriptor sets from a pool
    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9}};

    VkDescriptorPoolCreateInfo pool_create_info = {};
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.maxSets = 6;
    pool_create_info.poolSizeCount = pool_sizes.size();
    pool_create_info.pPoolSizes = pool_sizes.data();
    CHECK_VULKAN(vkCreateDescriptorPool(
        device->logical_device(), &pool_create_info, nullptr, &desc_pool));

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pSetLayouts = &volume_data_layout;
    alloc_info.descriptorPool = desc_pool;
    alloc_info.descriptorSetCount = 1;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &volume_data_desc_set));

    alloc_info.pSetLayouts = &compute_active_layout;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &compute_active_desc_set));

    alloc_info.pSetLayouts = &stream_compact_layout;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &stream_compact_desc_set));
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &stream_compact_remainder_desc_set));

    alloc_info.pSetLayouts = &compute_num_verts_layout;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &compute_num_verts_desc_set));

    alloc_info.pSetLayouts = &compute_verts_layout;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &compute_verts_desc_set));

    // Build the different pipeline layouts and pipelines
    vkrt::make_basic_compute_pipeline(compute_active_voxel_spv,
                                      sizeof(compute_active_voxel_spv),
                                      {volume_data_layout, compute_active_layout},
                                      *device,
                                      compute_active_pipeline_layout,
                                      compute_active_pipeline);

    vkrt::make_basic_compute_pipeline(stream_compact_spv,
                                      sizeof(stream_compact_spv),
                                      {stream_compact_layout},
                                      *device,
                                      stream_compact_pipeline_layout,
                                      stream_compact_pipeline);

    vkrt::make_basic_compute_pipeline(compute_num_verts_spv,
                                      sizeof(compute_num_verts_spv),
                                      {volume_data_layout, compute_num_verts_layout},
                                      *device,
                                      compute_num_verts_pipeline_layout,
                                      compute_num_verts_pipeline);

    vkrt::make_basic_compute_pipeline(compute_vertices_spv,
                                      sizeof(compute_vertices_spv),
                                      {volume_data_layout, compute_verts_layout},
                                      *device,
                                      compute_verts_pipeline_layout,
                                      compute_verts_pipeline);

    command_pool = device->make_command_pool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = command_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        CHECK_VULKAN(
            vkAllocateCommandBuffers(device->logical_device(), &info, &command_buffer));
    }
    {
        VkFenceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK_VULKAN(vkCreateFence(device->logical_device(), &info, nullptr, &fence));
    }

    size_t voxel_size = 0;
#if UINT8_VOLUME
    if (volume_type == "uint8") {
        voxel_size = 1;
    }
#endif
#if UINT16_VOLUME
    if (volume_type == "uint16") {
        voxel_size = 2;
    }
#endif
#if UINT32_VOLUME
    if (volume_type == "uint32") {
        voxel_size = 4;
    }
#endif
#if FLOAT32_VOLUME
    if (volume_type == "float32") {
        voxel_size = 4;
    }
#endif
    if (voxel_size == 0) {
        std::cout << "Volume type '" << volume_type
                  << "' support was not built, please recompile" << std::endl;
        throw std::runtime_error("Rebuild with " + volume_type + " support");
    }

    volume_buffer = vkrt::Buffer::device(
        *device,
        size_t(volume_dims.x) * size_t(volume_dims.y) * size_t(volume_dims.z) * voxel_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto volume_upload =
        vkrt::Buffer::host(*device, volume_buffer->size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    volume_info_buffer = vkrt::Buffer::device(
        *device,
        4 * 4 + 4,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    upload_volume_info_buffer = vkrt::Buffer::host(
        *device, volume_info_buffer->size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    {
        uint8_t *map = reinterpret_cast<uint8_t *>(upload_volume_info_buffer->map());
        glm::uvec3 *dims = reinterpret_cast<glm::uvec3 *>(map);
        *dims = volume_dims;
        float *isovalue = reinterpret_cast<float *>(map + sizeof(glm::uvec4));
        *isovalue = 128.f;
        upload_volume_info_buffer->unmap();
    }
    std::memcpy(volume_upload->map(), volume_data, volume_upload->size());
    volume_upload->unmap();

    tri_table_buffer = vkrt::Buffer::device(
        *device,
        256 * 16 * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    auto upload_tri_table_buffer =
        vkrt::Buffer::host(*device, 256 * 16 * sizeof(int), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(upload_tri_table_buffer->map(), tri_table, upload_tri_table_buffer->size());
    upload_tri_table_buffer->unmap();

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = upload_volume_info_buffer->size();
        vkCmdCopyBuffer(command_buffer,
                        upload_volume_info_buffer->handle(),
                        volume_info_buffer->handle(),
                        1,
                        &copy_cmd);
    }
    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = volume_upload->size();
        vkCmdCopyBuffer(
            command_buffer, volume_upload->handle(), volume_buffer->handle(), 1, &copy_cmd);
    }
    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = upload_tri_table_buffer->size();
        vkCmdCopyBuffer(command_buffer,
                        upload_tri_table_buffer->handle(),
                        tri_table_buffer->handle(),
                        1,
                        &copy_cmd);
    }
    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
    CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    const size_t voxels_to_process =
        size_t(volume_dims.x - 1) * size_t(volume_dims.y - 1) * size_t(volume_dims.z - 1);
    active_voxel_buffer = vkrt::Buffer::device(
        *device,
        voxels_to_process * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    active_voxel_offsets_buffer = vkrt::Buffer::device(
        *device,
        active_voxel_scanner.get_aligned_size(voxels_to_process) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    active_voxel_scanner.prepare_gpu_input(active_voxel_offsets_buffer, voxels_to_process);

    vkrt::DescriptorSetUpdater()
        .write_ssbo(volume_data_desc_set, 0, volume_buffer)
        .write_ubo(volume_data_desc_set, 1, volume_info_buffer)
        .write_ssbo(compute_active_desc_set, 0, active_voxel_buffer)
        .update(*device);
}

uint32_t MarchingCubes::compute_surface(const float isovalue)
{
    {
        uint8_t *map = reinterpret_cast<uint8_t *>(upload_volume_info_buffer->map());
        float *v = reinterpret_cast<float *>(map + sizeof(glm::uvec4));
        *v = isovalue;
        upload_volume_info_buffer->unmap();

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        VkBufferCopy copy_cmd = {};
        copy_cmd.size = upload_volume_info_buffer->size();
        vkCmdCopyBuffer(command_buffer,
                        upload_volume_info_buffer->handle(),
                        volume_info_buffer->handle(),
                        1,
                        &copy_cmd);
        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    }
    using namespace std::chrono;

    auto start = steady_clock::now();
    const uint32_t total_active = compute_active_voxels();
    auto end = steady_clock::now();
    std::cout << "compute_active_voxels took "
              << duration_cast<milliseconds>(end - start).count() << "\n";
    if (total_active == 0) {
        return 0;
    }

    start = steady_clock::now();
    auto active_voxel_ids = compact_active_voxels(total_active);
    end = steady_clock::now();
    std::cout << "compact_active_voxels took "
              << duration_cast<milliseconds>(end - start).count() << "\n";

    uint32_t total_vertices = 0;
    start = steady_clock::now();
    auto vertex_offset_buffer = compute_num_vertices(active_voxel_ids, total_vertices);
    end = steady_clock::now();
    std::cout << "compute_num_vertices took "
              << duration_cast<milliseconds>(end - start).count() << "\n";
    if (total_vertices == 0) {
        return 0;
    }
    start = steady_clock::now();
    compute_vertices(active_voxel_ids, vertex_offset_buffer, total_vertices);
    end = steady_clock::now();
    std::cout << "compute_vertices took " << duration_cast<milliseconds>(end - start).count()
              << "\n";
    return total_vertices;
}

uint32_t MarchingCubes::compute_active_voxels()
{
    const size_t voxels_to_process =
        size_t(volume_dims.x - 1) * size_t(volume_dims.y - 1) * size_t(volume_dims.z - 1);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_active_pipeline);

    const std::vector<VkDescriptorSet> desc_sets = {volume_data_desc_set,
                                                    compute_active_desc_set};
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_active_pipeline_layout,
                            0,
                            2,
                            desc_sets.data(),
                            0,
                            nullptr);
    vkCmdDispatch(command_buffer, volume_dims.x - 1, volume_dims.y - 1, volume_dims.z - 1);

    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = active_voxel_buffer->size();
        vkCmdCopyBuffer(command_buffer,
                        active_voxel_buffer->handle(),
                        active_voxel_offsets_buffer->handle(),
                        1,
                        &copy_cmd);
    }

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
    CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));
    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    return active_voxel_scanner.scan();
}

std::shared_ptr<vkrt::Buffer> MarchingCubes::compact_active_voxels(const uint32_t total_active)
{
    if (total_active > current_total_active) {
        active_voxel_ids = vkrt::Buffer::device(
            *device,
            total_active * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        current_total_active = total_active;
    }

    const size_t voxels_to_process =
        size_t(volume_dims.x - 1) * size_t(volume_dims.y - 1) * size_t(volume_dims.z - 1);

    // Note: not using push constants to send the offset to match the WebGPU version,
    // which currently does not have push constant support. Also following the 256b
    // dynamic offset restriction from Dawn to match closely
    const size_t num_chunks = static_cast<size_t>(
        std::ceil(static_cast<float>(voxels_to_process) / max_dispatch_size));
    auto chunk_offsets = vkrt::Buffer::device(
        *device,
        num_chunks * 256,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    vkrt::DescriptorSetUpdater()
        .write_ssbo_dynamic(stream_compact_desc_set,
                            0,
                            active_voxel_buffer,
                            0,
                            4 * std::min(voxels_to_process, max_dispatch_size))
        .write_ssbo_dynamic(stream_compact_desc_set,
                            1,
                            active_voxel_offsets_buffer,
                            0,
                            4 * std::min(voxels_to_process, max_dispatch_size))
        .write_ubo_dynamic(stream_compact_desc_set, 2, chunk_offsets, 0, 4)
        .write_ssbo(stream_compact_desc_set, 3, active_voxel_ids)
        .write_ssbo_dynamic(stream_compact_remainder_desc_set,
                            0,
                            active_voxel_buffer,
                            0,
                            4 * (voxels_to_process % max_dispatch_size))
        .write_ssbo_dynamic(stream_compact_remainder_desc_set,
                            1,
                            active_voxel_offsets_buffer,
                            0,
                            4 * (voxels_to_process % max_dispatch_size))
        .write_ubo_dynamic(stream_compact_remainder_desc_set, 2, chunk_offsets, 0, 4)
        .write_ssbo(stream_compact_remainder_desc_set, 3, active_voxel_ids)
        .update(*device);

    auto upload_chunks =
        vkrt::Buffer::host(*device, chunk_offsets->size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    uint32_t *offs = reinterpret_cast<uint32_t *>(upload_chunks->map());
    for (size_t i = 0; i < num_chunks; ++i) {
        offs[i * 64] = i * max_dispatch_size;
    }
    upload_chunks->unmap();

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = barrier.srcAccessMask;
    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = upload_chunks->size();
        vkCmdCopyBuffer(
            command_buffer, upload_chunks->handle(), chunk_offsets->handle(), 1, &copy_cmd);

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        const size_t num_work_groups =
            std::min(voxels_to_process - i * max_dispatch_size, max_dispatch_size);
        const std::vector<uint32_t> dynamic_offsets = {
            uint32_t(i * max_dispatch_size * sizeof(uint32_t)),
            uint32_t(i * max_dispatch_size * sizeof(uint32_t)),
            uint32_t(i * 256)};

        vkCmdBindPipeline(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stream_compact_pipeline);
        VkDescriptorSet desc_set = stream_compact_desc_set;
        if (num_work_groups < max_dispatch_size) {
            desc_set = stream_compact_remainder_desc_set;
        }
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                stream_compact_pipeline_layout,
                                0,
                                1,
                                &desc_set,
                                dynamic_offsets.size(),
                                dynamic_offsets.data());
        vkCmdDispatch(command_buffer, num_work_groups, 1, 1);

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);
    }

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    return active_voxel_ids;
}

std::shared_ptr<vkrt::Buffer> MarchingCubes::compute_num_vertices(
    std::shared_ptr<vkrt::Buffer> &active_voxel_ids, uint32_t &total_vertices)
{
    const size_t total_active = active_voxel_ids->size() / sizeof(uint32_t);
    const uint32_t aligned_total_active = num_verts_scanner.get_aligned_size(total_active);
    if (aligned_total_active > current_aligned_total_active) {
        num_verts_buffer = vkrt::Buffer::device(*device,
                                                aligned_total_active * sizeof(uint32_t),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        current_aligned_total_active = aligned_total_active;
    }

    vkrt::DescriptorSetUpdater()
        .write_ssbo(compute_num_verts_desc_set, 0, active_voxel_ids)
        .write_ssbo(compute_num_verts_desc_set, 1, num_verts_buffer)
        .write_ubo(compute_num_verts_desc_set, 2, tri_table_buffer)
        .update(*device);

    using namespace std::chrono;
    auto start = steady_clock::now();
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    vkCmdBindPipeline(
        command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_num_verts_pipeline);

    const std::vector<VkDescriptorSet> desc_sets = {volume_data_desc_set,
                                                    compute_num_verts_desc_set};
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_num_verts_pipeline_layout,
                            0,
                            2,
                            desc_sets.data(),
                            0,
                            nullptr);
    vkCmdDispatch(command_buffer, total_active, 1, 1);

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    auto end = steady_clock::now();
    std::cout << "num_verts pipeline only: "
              << duration_cast<milliseconds>(end - start).count() << "\n";

    num_verts_scanner.prepare_gpu_input(num_verts_buffer, total_active);
    total_vertices = num_verts_scanner.scan();
    return num_verts_buffer;
}

void MarchingCubes::compute_vertices(std::shared_ptr<vkrt::Buffer> &active_voxel_ids,
                                     std::shared_ptr<vkrt::Buffer> &vertex_offset_buffer,
                                     const uint32_t total_vertices)
{
    const size_t total_active = active_voxel_ids->size() / sizeof(uint32_t);

    if (!vertex_buffer || total_vertices * 16 > vertex_buffer->size()) {
        vertex_buffer = vkrt::Buffer::device(
            *device,
            total_vertices * 16,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    }

    vkrt::DescriptorSetUpdater()
        .write_ssbo(compute_verts_desc_set, 0, active_voxel_ids)
        .write_ssbo(compute_verts_desc_set, 1, vertex_offset_buffer)
        .write_ssbo(compute_verts_desc_set, 2, vertex_buffer)
        .write_ubo(compute_verts_desc_set, 3, tri_table_buffer)
        .update(*device);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_verts_pipeline);

    const std::vector<VkDescriptorSet> desc_sets = {volume_data_desc_set,
                                                    compute_verts_desc_set};
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_verts_pipeline_layout,
                            0,
                            2,
                            desc_sets.data(),
                            0,
                            nullptr);
    vkCmdDispatch(command_buffer, total_active, 1, 1);

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
}
