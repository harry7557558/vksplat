#pragma once

#include "gs_pipeline.h"

#include "perf_timer.h"
#include "scheduler.h"


PACK_STRUCT(struct VulkanGSRendererUniforms {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t num_splats;
    uint32_t active_sh;
    uint32_t step;
    uint32_t camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float world_view_transform[16];
});


class VulkanGSRenderer : public VulkanGSPipeline {
public:
    VulkanGSRenderer();
    ~VulkanGSRenderer();

    void initialize(const std::map<std::string, std::string> &spirv_paths, int device_id);
    void cleanup();

    void executeProjectionForward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, size_t alloc_reserve=0);
    void executeGenerateKeys(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeComputeTileRanges(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeRasterizeForward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeRasterizeBackward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);

    void executeCalculateIndexBufferOffset(VulkanGSPipelineBuffers& buffers);
    void executeSort(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, int num_bits);

private:
    enum class RasterBackwardImpl {
        PerPixel = 0,
        PerSplat = 1,
        Tensor_0_8_0 = 2,
        Tensor_0_8_8 = 3,
        Tensor_1_16_0 = 4,
        size = 5,
        Default = 0x3fffffff,
    };
    std::vector<RasterBackwardImpl> rasterizeBackwardAlternatives;
    ThompsonSamplingScheduler rasterizeBackwardScheduler;
    void initRasterizationBackwardScheduler();

protected:
    virtual DeviceRequirement getDeviceRequirement();

    void executeCumsum(
        VulkanGSPipelineBuffers &buffers,
        Buffer<int32_t> &input_buffer,
        Buffer<int32_t> &output_buffer
    );
    int32_t executeSum(
        VulkanGSPipelineBuffers &buffers,
        Buffer<int32_t> &input_buffer
    );
    void executeWhere(
        VulkanGSPipelineBuffers& buffers,
        Buffer<int32_t> &input_buffer,
        Buffer<int32_t> &output_buffer
    );

    _ComputePipeline pipeline_projection_forward = _ComputePipeline(11);
    _ComputePipeline pipeline_generate_keys = _ComputePipeline(7);
    _ComputePipeline pipeline_compute_tile_ranges[2] = {
        _ComputePipeline(2),
        _ComputePipeline(2)
    };
    _ComputePipelinePair pipeline_rasterize_forward = _ComputePipelinePair(7);
    _ComputePipelinePair pipeline_rasterize_backward[(size_t)RasterBackwardImpl::size] = {
        _ComputePipelinePair(11),  // per pixel
        _ComputePipelinePair(11),  // per splat
        _ComputePipelinePair(11),  // tensor
        _ComputePipelinePair(11),  // tensor
        _ComputePipelinePair(11),  // tensor
    };
    struct _CumsumComputePipeline {
        _ComputePipeline single_pass = _ComputePipeline(2);
        _ComputePipeline block_scan = _ComputePipeline(3);
        _ComputePipeline scan_block_sums = _ComputePipeline(3);
        _ComputePipeline add_block_offsets = _ComputePipeline(3);
    } pipeline_cumsum;
    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(3);
        _ComputePipeline spine = _ComputePipeline(2);
        _ComputePipeline downsweep = _ComputePipeline(6);
    } pipeline_sorting_1, pipeline_sorting_2;
    _ComputePipeline pipeline_sum = _ComputePipeline(2);
    _ComputePipeline pipeline_where = _ComputePipeline(3);
    _ComputePipeline pipeline_null = _ComputePipeline(0);

};
