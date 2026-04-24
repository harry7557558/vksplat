#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>  // memcpy
#include <mutex>
#include <cmath>

#include "config.h"

// https://stackoverflow.com/a/3312896
#ifdef __GNUC__
#define PACK_STRUCT( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#endif
#ifdef _MSC_VER
#define PACK_STRUCT( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif


// Buffers
struct _VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t allocSize;  // allocated size in bytes
    size_t size;  // actual size in bytes

    _VulkanBuffer()
        : buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE),
          allocSize(0), size(0) {}
    
    _VulkanBuffer(const _VulkanBuffer &other)
        : buffer(other.buffer), memory(other.memory),
          allocSize(other.allocSize), size(other.size) {}

    _VulkanBuffer operator=(const _VulkanBuffer &other) {
        memcpy((void*)this, &other, sizeof(_VulkanBuffer));
        return *this;
    }

    // used to test if descriptor needs to be updated
    bool operator==(const _VulkanBuffer &other) const {
        // printf("%d %d %d %d  ", int(buffer == other.buffer), int(memory == other.memory),
        //     int(allocSize == other.allocSize), int(size == other.size));
        // return buffer == other.buffer && memory == other.memory &&
        //        allocSize == other.allocSize && size == other.size;
        return buffer == other.buffer && memory == other.memory && allocSize == other.allocSize;
    }
};



template<typename T>
class Buffer : public std::vector<T> {
public:
    _VulkanBuffer deviceBuffer;

    Buffer() : std::vector<T>(), deviceBuffer() {}
    Buffer(const Buffer &other) : std::vector<T>(other), deviceBuffer(other.deviceBuffer) {}

    size_t byteLength() const { return this->size() * sizeof(T); }
    size_t deviceSize() const { return deviceBuffer.size / sizeof(T); }

};

#define _DECL_DEVICE_BUFFER(dtype, name) Buffer<dtype> name = Buffer<dtype>(false) 
// #define _DECL_DEVICE_BUFFER(dtype, name) Buffer<dtype> name



struct VulkanGSPipelineBuffers {
    size_t num_splats = 0;
    size_t num_indices = 0;

    // projection inputs
    Buffer<float> xyz_ws;  // (N, 3)
    Buffer<float> sh_coeffs;  // (N, 16, 3)
    Buffer<float> rotations;  // (N, 4)
    Buffer<float> scales_opacs;  // (N, 4)

    // projection outputs
    Buffer<int32_t> tiles_touched;  // (N,)
    Buffer<int64_t> rect_tile_space;  // (N,)
    Buffer<int32_t> radii;  // (N,)
    Buffer<float> xy_vs;  // (N, 2)
    Buffer<float> depths;  // (N, 1)
    Buffer<float> inv_cov_vs_opacity;  // (N, 4)
    Buffer<float> rgb;  // (N, 3)

    // tiles
    Buffer<int32_t> index_buffer_offset;  // N
    Buffer<sortingKey_t> sorting_keys_1;  // NInt [no_shrink]
    Buffer<sortingKey_t> sorting_keys_2;  // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_1;  // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_2;  // NInt [no_shrink]
    Buffer<int32_t> tile_ranges;  // (Gh*Gw, 2)
    bool is_unsorted_1 = true;
    Buffer<sortingKey_t>& unsorted_keys()
      { return is_unsorted_1 ? sorting_keys_1 : sorting_keys_2; }
    Buffer<sortingKey_t>& sorted_keys()
      { return is_unsorted_1 ? sorting_keys_2 : sorting_keys_1; }
    Buffer<sortingKey_t>& unsorted_gauss_idx()
      { return is_unsorted_1 ? sorting_gauss_idx_1 : sorting_gauss_idx_2; }
    Buffer<sortingKey_t>& sorted_gauss_idx()
      { return is_unsorted_1 ? sorting_gauss_idx_2 : sorting_gauss_idx_1; }

    // pixels
    Buffer<float> pixel_state;  // (H, W, 4)
    Buffer<int32_t> n_contributors;  // (H, W, 1)

    // rasterization v_outputs
    Buffer<float> ssim_map;  // (H, W, 3, 4)
    Buffer<float> v_pixel_state;  // (H, W, 4)
    Buffer<uint8_t> ref_image;  // (H, W, 4)

    // rasterization v_inputs, projection v_outputs
    Buffer<float> v_xy_vs;  // (N, 2)
    Buffer<float> v_depths;  // (N, 1)
    Buffer<float> v_inv_cov_vs_opacity;  // (N, 4)
    Buffer<float> v_rgb;  // (N, 3)

    // optimizer parameters (Adam)
    Buffer<float> g_xyz_ws;  // (N, 2, 3)
    Buffer<float> g_sh_coeffs_1;  // (N, 16, 3)
    Buffer<float> g_sh_coeffs_2;  // (N, 16, 3)
    Buffer<float> g_rotations;  // (N, 2, 4)
    Buffer<float> g_scales_opacs;  // (N, 2, 4)

    // default densification
    Buffer<float> default_grad;  // (N, 2), sum and count
    Buffer<float> default_radii;  // (N,)
    Buffer<int32_t> default_dupli_mask;  // (N,)
    Buffer<int32_t> default_split_mask;  // (N,)
    Buffer<int32_t> default_keep_mask;  // (N,)

    // MCMC
    Buffer<int32_t> mcmc_sample_probs;  // (N,)
    Buffer<int32_t> mcmc_sample_probs_cumsum;  // (N,)
    Buffer<int32_t> mcmc_index_map;  // (N,)
    Buffer<int32_t> mcmc_n_idx_buffer;  // (N,)

    // intermediate buffers
    Buffer<float> _temp_gauss_attr;
    Buffer<int32_t> _temp_indices;
    Buffer<int32_t> _temp_sum;
    Buffer<int32_t> _temp_cumsum;
    Buffer<int32_t> _cumsum_blockSums;
    Buffer<int32_t> _cumsum_blockSums2;
    Buffer<int32_t> _sorting_histogram;
    Buffer<int32_t> _sorting_histogram_cumsum;
    
    size_t getTotalAllocSize();
    void updateTotalAllocSize();
    size_t maxTotalAllocSize = 0;
    std::map<std::string, size_t> getVramBreakdown();

    template<typename T> static void reorderSH(Buffer<T> &coeffs);
    template<typename T> static void undoReorderSH(Buffer<T> &coeffs, size_t num_splats);

    static void assignScalesOpacs(Buffer<float> &scales_opacs, size_t n, const float* scales, const float* opacs);
};

float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);
