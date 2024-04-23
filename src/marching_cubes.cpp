#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include "compute_marching_cubes.h"
#include "exclusive_scan.h"
#include "vulkan_utils.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <volume.raw> <isovalue> [output.obj]\n";
        return 1;
    }

    const std::string file = argv[1];
    const float isovalue = std::stof(argv[2]);
    const std::regex match_filename("(\\w+)_(\\d+)x(\\d+)x(\\d+)_(.+)\\.raw");
    auto matches = std::sregex_iterator(file.begin(), file.end(), match_filename);
    if (matches == std::sregex_iterator() || matches->size() != 6) {
        std::cerr << "Unrecognized raw volume naming scheme, expected a format like: "
                  << "'<name>_<X>x<Y>x<Z>_<data type>.raw' but '" << file << "' did not match"
                  << std::endl;
        throw std::runtime_error("Invalaid raw file naming scheme");
    }
    std::string output;
    if (argc == 4) {
        output = argv[3];
    }

    const glm::uvec3 volume_dims(
        std::stoi((*matches)[2]), std::stoi((*matches)[3]), std::stoi((*matches)[4]));
    const std::string volume_type = (*matches)[5];

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
    const size_t volume_bytes =
        size_t(volume_dims.x) * size_t(volume_dims.y) * size_t(volume_dims.z) * voxel_size;
    std::vector<uint8_t> volume_data(volume_bytes, 0);
    std::ifstream fin(file.c_str(), std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open " << file << "\n";
        return 1;
    }
    if (!fin.read(reinterpret_cast<char *>(volume_data.data()), volume_bytes)) {
        std::cerr << "Failed to read volume data\n";
        return 1;
    }

    std::shared_ptr<vkrt::Device> device = std::make_shared<vkrt::Device>();

    MarchingCubes marching_cubes(device, volume_data.data(), volume_dims, volume_type);

    using namespace std::chrono;
    uint32_t num_verts = 0;
    size_t total_time = 0;
    const size_t num_iters = 10;
    for (size_t i = 0; i < num_iters; ++i) {
        auto start = steady_clock::now();
        num_verts = marching_cubes.compute_surface(isovalue);
        auto end = steady_clock::now();
        std::cout << "Extraction of surface w/ " << num_verts / 3 << " triangles took "
                  << duration_cast<milliseconds>(end - start).count() << "ms\n";
        total_time += duration_cast<milliseconds>(end - start).count();
    }
    std::cout << "Avg. time " << static_cast<float>(total_time) / num_iters << "ms\n";

    if (num_verts == 0 || output.empty()) {
        return 0;
    }

    VkCommandPool command_pool =
        device->make_command_pool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    VkCommandBuffer command_buffer;
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = command_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        CHECK_VULKAN(
            vkAllocateCommandBuffers(device->logical_device(), &info, &command_buffer));
    }

    auto readback_verts = vkrt::Buffer::host(
        *device, marching_cubes.vertex_buffer->size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    {
        VkBufferCopy copy_cmd = {};
        copy_cmd.size = marching_cubes.vertex_buffer->size();
        vkCmdCopyBuffer(command_buffer,
                        marching_cubes.vertex_buffer->handle(),
                        readback_verts->handle(),
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

    glm::vec4 *vertices = reinterpret_cast<glm::vec4 *>(readback_verts->map());
    std::ofstream fout(output.c_str());
    fout << "# Isosurface of " << file << " at isovalue " << isovalue << " (" << num_verts / 3
         << " triangles)\n";
    for (size_t i = 0; i < num_verts; ++i) {
        fout << "v " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << "\n";
    }
    for (size_t i = 0; i < num_verts; i += 3) {
        fout << "f " << i + 1 << " " << i + 2 << " " << i + 3 << "\n";
    }

    readback_verts->unmap();

    return 0;
}
