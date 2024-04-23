#include "exclusive_scan.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include "scan_shaders_embedded_spv.h"
#include "util.h"

const size_t ExclusiveScanner::block_size = BLOCK_SIZE;
const size_t ExclusiveScanner::workgroup_size = BLOCK_SIZE / 2;
const size_t ExclusiveScanner::max_scan_size = BLOCK_SIZE * BLOCK_SIZE;

ExclusiveScanner::ExclusiveScanner(std::shared_ptr<vkrt::Device> &device) : device(device)
{
    // Make descriptor sets
    scan_blocks_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(
                0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    scan_block_results_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
            .build(*device);

    // Make pipelines for scan/add blocks pipelines
    vkrt::make_basic_compute_pipeline(prefix_sum_spv,
                                      sizeof(prefix_sum_spv),
                                      {scan_blocks_layout},
                                      *device,
                                      scan_blocks_pipeline_layout,
                                      scan_blocks_pipeline);

    vkrt::make_basic_compute_pipeline(add_block_sums_spv,
                                      sizeof(add_block_sums_spv),
                                      {scan_blocks_layout},
                                      *device,
                                      scan_blocks_pipeline_layout,
                                      add_block_sums_pipeline);

    vkrt::make_basic_compute_pipeline(block_prefix_sum_spv,
                                      sizeof(block_prefix_sum_spv),
                                      {scan_block_results_layout},
                                      *device,
                                      scan_block_results_pipeline_layout,
                                      scan_block_results_pipeline);

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 2},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};

    VkDescriptorPoolCreateInfo pool_create_info = {};
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.maxSets = 3;
    pool_create_info.poolSizeCount = pool_sizes.size();
    pool_create_info.pPoolSizes = pool_sizes.data();
    CHECK_VULKAN(vkCreateDescriptorPool(
        device->logical_device(), &pool_create_info, nullptr, &desc_pool));

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pSetLayouts = &scan_blocks_layout;
    alloc_info.descriptorPool = desc_pool;
    alloc_info.descriptorSetCount = 1;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &scan_blocks_desc_set));
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &scan_blocks_remainder_desc_set));

    alloc_info.pSetLayouts = &scan_block_results_layout;
    CHECK_VULKAN(vkAllocateDescriptorSets(
        device->logical_device(), &alloc_info, &scan_block_results_desc_set));

    readback_buffer = vkrt::Buffer::host(*device, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    block_sum_buffer = vkrt::Buffer::device(
        *device,
        block_size * 4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    carry_buffer = vkrt::Buffer::device(*device,
                                        8,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    clearcarry_buffer = vkrt::Buffer::host(*device, 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memset(clearcarry_buffer->map(), 0, clearcarry_buffer->size());
    clearcarry_buffer->unmap();

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
}

size_t ExclusiveScanner::get_aligned_size(size_t size)
{
    return align_to(size, block_size);
}

void ExclusiveScanner::prepare_input(const std::vector<uint32_t> &array)
{
    auto upload_input = vkrt::Buffer::host(
        *device, array.size() * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(upload_input->map(), array.data(), upload_input->size());
    upload_input->unmap();

    auto gpu_input = vkrt::Buffer::device(*device,
                                          get_aligned_size(array.size()) * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    VkBufferCopy copy_cmd = {};
    copy_cmd.size = upload_input->size();
    vkCmdCopyBuffer(command_buffer, upload_input->handle(), gpu_input->handle(), 1, &copy_cmd);
    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
    CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    prepare_gpu_input(gpu_input, array.size());
}

void ExclusiveScanner::prepare_gpu_input(std::shared_ptr<vkrt::Buffer> &buffer,
                                         size_t data_size)
{
    if (get_aligned_size(buffer->size()) != buffer->size()) {
        throw std::runtime_error(
            "Buffer size must be aligned via ExclusiveScanner::get_aligned_size");
    }

    input_buffer = buffer;
    const size_t scan_num_elements = buffer->size() / 4;

    // Write the descriptor sets (aka, the WebGPU bind groups)
    vkrt::DescriptorSetUpdater()
        .write_ssbo_dynamic(scan_blocks_desc_set,
                            0,
                            input_buffer,
                            0,
                            4 * std::min(max_scan_size, scan_num_elements))
        .write_ssbo(scan_blocks_desc_set, 1, block_sum_buffer)
        .write_ssbo_dynamic(scan_blocks_remainder_desc_set,
                            0,
                            input_buffer,
                            0,
                            4 * (scan_num_elements % max_scan_size))
        .write_ssbo(scan_blocks_remainder_desc_set, 1, block_sum_buffer)
        .write_ssbo(scan_block_results_desc_set, 0, block_sum_buffer)
        .write_ssbo(scan_block_results_desc_set, 1, carry_buffer)
        .update(*device);

    const size_t num_chunks = std::ceil(static_cast<float>(scan_num_elements) / max_scan_size);
    std::vector<uint32_t> offsets;
    for (size_t i = 0; i < num_chunks; ++i) {
        offsets.push_back(i * max_scan_size * 4);
    }

    // Build the command buffer
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = 0;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));
    // Clear the carry buffer and readback sum entry
    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = clearcarry_buffer->size();
        vkCmdCopyBuffer(
            command_buffer, clearcarry_buffer->handle(), carry_buffer->handle(), 1, &copy_cmd);
    }
    if (data_size < scan_num_elements) {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = 4;
        copy_cmd.dstOffset = data_size * 4;
        vkCmdCopyBuffer(
            command_buffer, clearcarry_buffer->handle(), input_buffer->handle(), 1, &copy_cmd);
    }

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = barrier.srcAccessMask;
    for (size_t i = 0; i < num_chunks; ++i) {
        const uint32_t num_work_groups =
            std::min((scan_num_elements - i * max_scan_size) / block_size, size_t(block_size));
        vkCmdBindPipeline(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_blocks_pipeline);

        VkDescriptorSet scan_desc_set = scan_blocks_desc_set;
        if (num_work_groups < max_scan_size / block_size) {
            scan_desc_set = scan_blocks_remainder_desc_set;
        }

        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                scan_blocks_pipeline_layout,
                                0,
                                1,
                                &scan_desc_set,
                                1,
                                &offsets[i]);
        vkCmdDispatch(command_buffer, num_work_groups, 1, 1);
        // Queue a barrier for the pass to finish
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

        vkCmdBindPipeline(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_block_results_pipeline);
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                scan_block_results_pipeline_layout,
                                0,
                                1,
                                &scan_block_results_desc_set,
                                0,
                                nullptr);
        vkCmdDispatch(command_buffer, 1, 1, 1);
        // Queue a barrier for the pass to finish
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

        vkCmdBindPipeline(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, add_block_sums_pipeline);
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                scan_blocks_pipeline_layout,
                                0,
                                1,
                                &scan_desc_set,
                                1,
                                &offsets[i]);
        vkCmdDispatch(command_buffer, num_work_groups, 1, 1);
        // Queue a barrier for the pass to finish
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

        // Update the carry buffer
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = 4;
        copy_cmd.srcOffset = 4;
        copy_cmd.dstOffset = 0;
        vkCmdCopyBuffer(
            command_buffer, carry_buffer->handle(), carry_buffer->handle(), 1, &copy_cmd);
    }
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
    // Readback the last element to return the total sum
    if (data_size < scan_num_elements) {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = 4;
        copy_cmd.srcOffset = data_size * 4;
        vkCmdCopyBuffer(
            command_buffer, input_buffer->handle(), readback_buffer->handle(), 1, &copy_cmd);
    } else {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = 4;
        copy_cmd.srcOffset = 4;
        vkCmdCopyBuffer(
            command_buffer, carry_buffer->handle(), readback_buffer->handle(), 1, &copy_cmd);
    }
    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));
}

uint32_t ExclusiveScanner::scan()
{
    using namespace std::chrono;
    auto start = steady_clock::now();
    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
    auto end = steady_clock::now();
    std::cout << "Scan took " << duration_cast<milliseconds>(end - start).count() << "ms\n";

    uint32_t sum = 0;
    std::memcpy(&sum, readback_buffer->map(), 4);
    readback_buffer->unmap();

    return sum;
}

uint32_t serial_exclusive_scan(const std::vector<uint32_t> &input,
                               std::vector<uint32_t> &output)
{
    output.resize(input.size(), 0);
    output[0] = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = input[i - 1] + output[i - 1];
    }
    return output[output.size() - 1] + input[input.size() - 1];
}
