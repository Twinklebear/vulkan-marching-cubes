#pragma once

#include <memory>
#include "vulkan_utils.h"

struct ExclusiveScanner {
    static const size_t block_size;
    static const size_t workgroup_size;
    static const size_t max_scan_size;

    std::shared_ptr<vkrt::Device> device;
    std::shared_ptr<vkrt::Buffer> input_buffer, block_sum_buffer, readback_buffer,
        carry_buffer, clearcarry_buffer;

    VkDescriptorSetLayout scan_blocks_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout scan_block_results_layout = VK_NULL_HANDLE;

    VkPipelineLayout scan_blocks_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout scan_block_results_pipeline_layout = VK_NULL_HANDLE;

    VkPipeline scan_blocks_pipeline = VK_NULL_HANDLE;
    VkPipeline scan_block_results_pipeline = VK_NULL_HANDLE;
    VkPipeline add_block_sums_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    VkDescriptorSet scan_blocks_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet scan_blocks_remainder_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet scan_block_results_desc_set = VK_NULL_HANDLE;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    VkFence fence = VK_NULL_HANDLE;

    ExclusiveScanner(std::shared_ptr<vkrt::Device> &device);

    size_t get_aligned_size(size_t size);

    void prepare_input(const std::vector<uint32_t> &array);

    void prepare_gpu_input(std::shared_ptr<vkrt::Buffer> &buffer, size_t data_size);

    uint32_t scan();
};

uint32_t serial_exclusive_scan(const std::vector<uint32_t> &input,
                               std::vector<uint32_t> &output);
