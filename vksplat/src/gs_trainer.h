#pragma once

#include "gs_renderer.h"

#include "colmap_reader.h"

#include <random>


// see simple_trainer.py for details
struct TrainerConfig {

    std::string output_dir;
    std::string output_ply;

    // dataset
    std::string dataset_dir;
    std::string image_dir;
    std::string mask_dir;
    std::string sparse_dir;
    int eval_interval;

    enum class CacheImage {
        CPU, GPU
    } image_cache_device;

    float global_scale;
    float init_scale;
    float init_opacity;

    enum class Strategy {
        Default, MCMC,
    } strategy;

    // optimizer
    int max_steps;
    float ssim_lambda;
    float means_lr;
    float means_lr_final;
    float features_dc_lr;
    float features_rest_lr;
    float opacities_lr;
    float scales_lr;
    float quats_lr;
    float scale_reg;
    float opacity_reg;

    inline float get_means_lr(int step, float scene_scale) const {
        float progress = fmin(float(step)/float(this->max_steps), 1.0f);
        return this->means_lr * pow(this->means_lr_final/this->means_lr, progress) * scene_scale;
    }

    // strategy
    int refine_start_iter;
    int refine_stop_iter;
    int refine_every;

    // default strategy
    float prune_opa;
    float grow_grad2d;
    float grow_scale3d;
    float grow_scale2d;
    float prune_scale3d;
    float prune_scale2d;
    int refine_scale2d_stop_iter;
    int reset_every;
    int stop_reset_at;
    int pause_refine_after_reset;

    // MCMC strategy
    float noise_lr;
    float min_opacity;
    float grow_factor;
    int cap_max;
};



class VulkanGSTrainer : public VulkanGSRenderer {
public:
    VulkanGSTrainer();
    ~VulkanGSTrainer();

    struct Camera {
        // https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
        enum Model {
            SIMPLE_PINHOLE = 0,  // f cx cy
            PINHOLE = 1,  // fx fy cx cy
            SIMPLE_RADIAL = 2,  // f cx cy k
            RADIAL = 3,  // f cx cy k1 k2
            OPENCV = 4,  // fx fy cx cy k1 k2 p1 p2
            OPENCV_FISHEYE = 5,  // fx fy cx cy k1 k2 k3 k4
            FULL_OPENCV = 6,  // fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
            FOV = 7,  // fx fy cx cy omega
            SIMPLE_RADIAL_FISHEYE = 8,  // f cx cy k
            RADIAL_FISHEYE = 9,  // f cx cy k1 k2
            THIN_PRISM_FISHEYE = 10,  // fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1
            RAD_TAN_THIN_PRISM_FISHEYE = 11,  // fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3
        } model;
        int h, w;
        float fx, fy, cx, cy;
        float k1, k2, p1, p2, k3, k4, k5, k6, sx1, sy1;
        glm::mat4 world_view_transform;
    };

    struct DatasetImage {
        std::string image_path;
        std::string mask_path;
        Camera camera;
        Buffer<uint8_t> buffer;  // (h, w, 4)
    };

    void initialize(const std::map<std::string, std::string> &spirv_paths, int device_id);
    void cleanup();

    void load_colmap_dataset(const TrainerConfig &config, VulkanGSPipelineBuffers& buffers);

    size_t num_train() const { return dataset_train.size(); }
    size_t num_val() const { return dataset_val.size(); }

    DatasetImage& get_train_image(size_t idx) { return dataset_train[idx]; }
    DatasetImage& get_val_image(size_t idx) { return dataset_val[idx]; }
    glm::mat4 get_dataparser_transform() const { return dataparser_transform; }

    void camera_to_uniforms(const Camera &cam, VulkanGSRendererUniforms &uniforms) const;
    void get_train_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const;
    void get_val_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const;

    void executeComputeSSIMGradient(const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, size_t train_idx);
    void executeFusedProjectionBackwardOptimizerStep(const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, int step);

    void executeDefaultPostBackward(const TrainerConfig &config, const VulkanGSRendererUniforms& renderer_uniforms, VulkanGSPipelineBuffers& buffers, int step);
    void executeMCMCPostBackward(const TrainerConfig &config, const VulkanGSRendererUniforms& renderer_uniforms, VulkanGSPipelineBuffers& buffers, int step);
    void executeMortonSorting(const VulkanGSRendererUniforms &uniforms, VulkanGSPipelineBuffers& buffers);

    void writePLY(std::string filename, VulkanGSPipelineBuffers& buffers);

private:
    std::default_random_engine rng;

    std::vector<DatasetImage> dataset_train;
    std::vector<DatasetImage> dataset_val;
    glm::mat4 dataparser_transform;
    float scene_scale;

    _ComputePipeline pipeline_ssim_forward = _ComputePipeline(3);
    _ComputePipeline pipeline_ssim_backward = _ComputePipeline(4);
    _ComputePipeline pipeline_fused_projection_backward_optimizer = _ComputePipeline(13);
    struct _DefaultComputePipeline {
        _ComputePipeline update_state = _ComputePipeline(4);
        _ComputePipeline compute_grow_mask = _ComputePipeline(5);
        _ComputePipeline duplicate = _ComputePipeline(12);
        _ComputePipeline split = _ComputePipeline(12);
        _ComputePipeline compute_prune_mask = _ComputePipeline(3);
        _ComputePipeline prune = _ComputePipeline(4);
        _ComputePipeline prune_mean = _ComputePipeline(4);
        _ComputePipeline prune_sh = _ComputePipeline(4);
        _ComputePipeline reset_opa = _ComputePipeline(2);
    } pipeline_default;
    struct _MCMCComputePipeline {
        _ComputePipeline inject_noise = _ComputePipeline(4);
        _ComputePipeline compute_probs = _ComputePipeline(2);
        _ComputePipeline compute_relocation_index_map = _ComputePipeline(4);
        _ComputePipeline compute_relocation = _ComputePipeline(7);
        _ComputePipeline update_relocation = _ComputePipeline(10);
        _ComputePipeline compute_add_index_map = _ComputePipeline(3);
        _ComputePipeline compute_add = _ComputePipeline(2);
        _ComputePipeline update_add = _ComputePipeline(10);
    } pipeline_mcmc;
    struct _MortonSortComputePipeline {
        _ComputePipeline compute_stats = _ComputePipeline(2);
        _ComputePipeline generate_keys = _ComputePipeline(4);
        _ComputePipeline apply_indices = _ComputePipeline(3);
        _ComputePipeline apply_indices_sh = _ComputePipeline(3);
        _ComputePipeline update_buffer = _ComputePipeline(2);
        _ComputePipeline update_buffer_sh = _ComputePipeline(2);
    } pipeline_morton_sort;

    void barrierAllGaussParams(VulkanGSPipelineBuffers& buffers);

};

