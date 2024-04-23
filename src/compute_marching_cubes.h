#pragma once

#include <memory>
#include "exclusive_scan.h"
#include "vulkan_utils.h"

struct MarchingCubes {
    static const size_t max_dispatch_size;

    std::shared_ptr<vkrt::Device> device;

    ExclusiveScanner active_voxel_scanner, num_verts_scanner;

    glm::uvec3 volume_dims;

    std::shared_ptr<vkrt::Buffer> volume_info_buffer, upload_volume_info_buffer, volume_buffer,
        active_voxel_buffer, active_voxel_offsets_buffer, vertex_buffer, tri_table_buffer;

    // Scratch buffers which we keep and re-use if the new computation
    // fits in the same buffer space
    std::shared_ptr<vkrt::Buffer> active_voxel_ids, num_verts_buffer;

    uint32_t current_total_active = 0;
    uint32_t current_aligned_total_active = 0;

    VkDescriptorSetLayout volume_data_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout compute_active_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout stream_compact_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout compute_num_verts_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout compute_verts_layout = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    VkDescriptorSet volume_data_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet compute_active_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet stream_compact_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet stream_compact_remainder_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet compute_num_verts_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet compute_verts_desc_set = VK_NULL_HANDLE;

    VkPipelineLayout compute_active_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout stream_compact_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout compute_num_verts_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout compute_verts_pipeline_layout = VK_NULL_HANDLE;

    VkPipeline compute_active_pipeline = VK_NULL_HANDLE;
    VkPipeline stream_compact_pipeline = VK_NULL_HANDLE;
    VkPipeline compute_num_verts_pipeline = VK_NULL_HANDLE;
    VkPipeline compute_verts_pipeline = VK_NULL_HANDLE;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    MarchingCubes(std::shared_ptr<vkrt::Device> &device,
                  void *volume_data,
                  const glm::uvec3 &volume_dims,
                  const std::string &volume_type);

    uint32_t compute_surface(const float isovalue);

private:
    uint32_t compute_active_voxels();

    std::shared_ptr<vkrt::Buffer> compact_active_voxels(const uint32_t total_active);

    std::shared_ptr<vkrt::Buffer> compute_num_vertices(
        std::shared_ptr<vkrt::Buffer> &active_voxel_ids, uint32_t &total_vertices);

    void compute_vertices(std::shared_ptr<vkrt::Buffer> &active_voxel_ids,
                          std::shared_ptr<vkrt::Buffer> &vertex_offset_buffer,
                          const uint32_t total_vertices);
};
