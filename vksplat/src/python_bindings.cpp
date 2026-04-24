#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gs_renderer.h"
#include "buffer.h"
#include "gs_trainer.h"

#include <chrono>

namespace py = pybind11;

class VkSplat {
public:
    // VulkanGSRenderer trainer;
    VulkanGSTrainer trainer;
    VulkanGSRendererUniforms uniforms;
    VulkanGSPipelineBuffers buffers;
    TrainerConfig config;

    template<typename Ti, typename To, bool on_device=true>
    py::array_t<To> buffer_to_array(
        Buffer<Ti> &buffer,
        size_t n1=(size_t)(-1),
        size_t n2=(size_t)(-1),
        std::string reorder = "",
        size_t n1_actual=(size_t)-1,
        size_t n1_offset=(size_t)0
    ) {
        if (on_device)
            trainer.copyFromDevice(buffer);
        if (reorder == "sh")
            buffers.undoReorderSH(buffer, buffers.num_splats);

        size_t n0 = buffer.size();
        if (n1 != (size_t)(-1)) n0 /= n1;
        if (n2 != (size_t)(-1)) n0 /= n2;

        std::vector<To> buffer_float;
        auto toFloat = [&](Ti* buffer_half) -> To* {
            size_t n = buffer.size();
            buffer_float.resize(n);
            for (size_t i = 0; i < n; i++)
                buffer_float[i] = (To)halfToFloat((uint16_t)buffer_half[i]);
            return buffer_float.data();
        };

        if (n1 == (size_t)(-1) && n2 == (size_t)(-1)) {
            typedef std::array<py::ssize_t, 1> dim;
            return py::array_t<To>(
                dim({ (py::ssize_t)n0 }),
                dim({ sizeof(To) }),
                sizeof(Ti) == sizeof(To) ? (To*)buffer.data() : toFloat(buffer.data())
            );
        }
        else if (n2 == (size_t)(-1)) {
            typedef std::array<py::ssize_t, 2> dim2;
            if (n1_actual == (size_t)-1)
                n1_actual = n1;
            return py::array_t<To>(
                dim2({ (py::ssize_t)n0, (py::ssize_t)n1_actual }),
                dim2({ (py::ssize_t)(n1*sizeof(To)), sizeof(To) }),
                sizeof(Ti) == sizeof(To) ? n1_offset+(To*)buffer.data() : toFloat(n1_offset+buffer.data())
            );
        }
        else {
            typedef std::array<py::ssize_t, 3> dim3;
            return py::array_t<To>(
                dim3({ (py::ssize_t)n0, (py::ssize_t)n1, (py::ssize_t)n2 }),
                dim3({ (py::ssize_t)(n1*n2*sizeof(To)), (py::ssize_t)(n2*sizeof(To)), sizeof(To) }),
                sizeof(Ti) == sizeof(To) ? (To*)buffer.data() : toFloat(buffer.data())
            );
        }
    }

    template<typename T>
    void array_to_buffer(
        py::array_t<T> array,
        Buffer<T> &buffer
    ) {
        auto array_buf = array.request();
        buffer.assign((T*)(array_buf.ptr), (T*)(array_buf.ptr) + array_buf.size);
        trainer.copyToDevice(buffer);
    }

public:

    void initialize(std::string spirv_dir, int device_id) {
        buffers.num_splats = 0;

        std::vector<std::string> spirv_paths = {
            "projection_forward",
            "generate_keys",
            "compute_tile_ranges",
            "rasterize_forward",
            "rasterize_backward_0",
            "rasterize_backward_1",
            "rasterize_backward_2",
            "rasterize_backward_3",
            "rasterize_backward_4",
            "cumsum_single_pass",
            "cumsum_block_scan",
            "cumsum_scan_block_sums",
            "cumsum_add_block_offsets",
            "radix_sort/upsweep",
            "radix_sort/spine",
            "radix_sort/downsweep",
            "ssim_forward",
            "ssim_backward",
            "fused_projection_backward_optimizer",
            "sum",
            "where",
            "default_update_state",
            "default_compute_grow_mask",
            "default_duplicate",
            "default_split",
            "default_compute_prune_mask",
            "default_prune",
            "default_prune_mean",
            "default_prune_sh",
            "default_reset_opa",
            "mcmc_inject_noise",
            "mcmc_compute_probs",
            "mcmc_compute_relocation_index_map",
            "mcmc_compute_relocation",
            "mcmc_update_relocation",
            "mcmc_compute_add_index_map",
            "mcmc_compute_add",
            "mcmc_update_add",
            "morton_sort_compute_stats",
            "morton_sort_generate_keys",
            "morton_sort_apply_indices",
            "morton_sort_apply_indices_sh",
            "morton_sort_update_buffer",
            "morton_sort_update_buffer_sh",
        };
        std::map<std::string, std::string> spirv_paths_dict;
        for (std::string name : spirv_paths) {
            spirv_paths_dict[name] = 
                (name.find('/') == std::string::npos) ?
                spirv_dir + "generated/" + name + ".spv" :
                spirv_dir + name + ".spv";
        }

        trainer.initialize(spirv_paths_dict, device_id);

        PerfTimer::reset();
    }

    py::dict set_train_config(py::dict train_config) {

        config.output_dir = train_config["output_dir"].cast<std::string>();
        config.output_ply = train_config["output_ply"].cast<std::string>();

        // dataset
        config.dataset_dir = train_config["dataset_dir"].cast<std::string>();
        config.image_dir = train_config["image_dir"].cast<std::string>();
        config.mask_dir = train_config["mask_dir"].cast<std::string>();
        config.sparse_dir = train_config["sparse_dir"].cast<std::string>();
        config.eval_interval = train_config["eval_interval"].cast<int>();

        std::string image_cache_device = train_config["image_cache_device"].cast<std::string>();
        if (image_cache_device == "cpu")
            config.image_cache_device = TrainerConfig::CacheImage::CPU;
        else if (image_cache_device == "gpu")
            config.image_cache_device = TrainerConfig::CacheImage::GPU;
        else throw std::runtime_error("Unknow image_cache_device: " + image_cache_device);

        config.global_scale = train_config["global_scale"].cast<float>();
        config.init_scale = train_config["init_scale"].cast<float>();
        config.init_opacity = train_config["init_opacity"].cast<float>();

        std::string strategy = train_config["strategy"].cast<std::string>();
        if (strategy == "default")
            config.strategy = TrainerConfig::Strategy::Default;
        else if (strategy == "mcmc")
            config.strategy = TrainerConfig::Strategy::MCMC;
        else throw std::runtime_error("Unknow strategy: " + strategy);

        // optimizer
        config.max_steps = train_config["max_steps"].cast<int>();
        config.ssim_lambda = train_config["ssim_lambda"].cast<float>();
        config.means_lr = train_config["means_lr"].cast<float>();
        config.means_lr_final = train_config["means_lr_final"].cast<float>();
        config.features_dc_lr = train_config["features_dc_lr"].cast<float>();
        config.features_rest_lr = train_config["features_rest_lr"].cast<float>();
        config.opacities_lr = train_config["opacities_lr"].cast<float>();
        config.scales_lr = train_config["scales_lr"].cast<float>();
        config.quats_lr = train_config["quats_lr"].cast<float>();
        config.scale_reg = train_config["scale_reg"].cast<float>();
        config.opacity_reg = train_config["opacity_reg"].cast<float>();

        // strategy
        config.refine_start_iter = train_config["refine_start_iter"].cast<int>();
        config.refine_stop_iter = train_config["refine_stop_iter"].cast<int>();
        config.refine_every = train_config["refine_every"].cast<int>();

        // default strategy
        config.prune_opa = train_config["prune_opa"].cast<float>();
        config.grow_grad2d = train_config["grow_grad2d"].cast<float>();
        config.grow_scale3d = train_config["grow_scale3d"].cast<float>();
        config.grow_scale2d = train_config["grow_scale2d"].cast<float>();
        config.prune_scale3d = train_config["prune_scale3d"].cast<float>();
        config.prune_scale2d = train_config["prune_scale2d"].cast<float>();
        config.refine_scale2d_stop_iter = train_config["refine_scale2d_stop_iter"].cast<int>();
        config.reset_every = train_config["reset_every"].cast<int>();
        config.stop_reset_at = train_config["stop_reset_at"].cast<int>();
        config.pause_refine_after_reset = train_config["pause_refine_after_reset"].cast<int>();

        // MCMC strategy
        config.noise_lr = train_config["noise_lr"].cast<float>();
        config.min_opacity = train_config["min_opacity"].cast<float>();
        config.grow_factor = train_config["grow_factor"].cast<float>();
        config.cap_max = train_config["cap_max"].cast<int>();

        // load dataset
        trainer.load_colmap_dataset(config, buffers);
        trainer.get_train_camera(0, uniforms);
        uniforms.active_sh = 3;

        // return metadata
        py::dict train_meta;
        glm::mat4 dataparser_transform = trainer.get_dataparser_transform();
        train_meta["dataparser_transform"] = py::array_t<float>(
            {4, 4},
            {sizeof(float), 4 * sizeof(float)},   // column-major
            reinterpret_cast<float*>(&dataparser_transform)
        );
        train_meta["device"] = trainer.get_device_info();
        return train_meta;
    }

    size_t num_train() { return trainer.num_train(); }
    size_t num_val() { return trainer.num_val(); }

    py::array_t<uint8_t> get_train_image(size_t &idx) {
        auto& im = trainer.get_train_image(idx);
        return buffer_to_array<uint8_t, uint8_t, false>(im.buffer, im.camera.w, 4);
    }
    py::array_t<uint8_t> get_val_image(size_t &idx) {
        auto& im = trainer.get_val_image(idx);
        return buffer_to_array<uint8_t, uint8_t, false>(im.buffer, im.camera.w, 4);
    }

    void set_uniforms(
        uint32_t active_sh,
        py::array_t<float> world_view_transform,
        uint32_t image_height, uint32_t image_width,
        float fx, float fy, float cx, float cy,
        bool is_fisheye
    ) {
        auto transform_buf = world_view_transform.request();
        if (transform_buf.ndim != 2 || transform_buf.shape[0] != 4 || transform_buf.shape[1] != 4)
            throw std::runtime_error("world_view_transform must be (4, 4) array");

        uniforms.active_sh = active_sh;
        uniforms.image_height = image_height;
        uniforms.image_width = image_width;
        uniforms.camera_model = is_fisheye ? 2 : 0;
        uniforms.fx = fx;
        uniforms.fy = fy;
        uniforms.cx = cx;
        uniforms.cy = cy;
        uniforms.grid_height = _CEIL_DIV(image_height, TILE_HEIGHT);
        uniforms.grid_width = _CEIL_DIV(image_width, TILE_WIDTH);

        // row/column major matrix conversion
        const float* transform_ptr = static_cast<float*>(transform_buf.ptr);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                uniforms.world_view_transform[4*i+j] = transform_ptr[4*j+i];

        uniforms.step = 0;
    }

    py::dict get_uniforms() {
        py::dict result;
        
        result["active_sh"] = uniforms.active_sh + 0;
        result["image_height"] = uniforms.image_height + 0;
        result["image_width"] = uniforms.image_width + 0;
        result["fx"] = uniforms.fx * 1.0f;
        result["fy"] = uniforms.fy * 1.0f;
        result["cx"] = uniforms.cx * 1.0f;
        result["cy"] = uniforms.cy * 1.0f;
        
        auto world_view_array = py::array_t<float>({4, 4});
        auto world_view_buf = world_view_array.request();
        float* world_view_ptr = static_cast<float*>(world_view_buf.ptr);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                world_view_ptr[4*i+j] = uniforms.world_view_transform[4*j+i];
        result["world_view_transform"] = world_view_array;
        
        auto K_array = py::array_t<float>({3, 3});
        auto K_buf = K_array.request();
        float* K_ptr = static_cast<float*>(K_buf.ptr);
        std::fill(K_ptr, K_ptr + 9, 0.0f);
        K_ptr[0] = uniforms.fx;  // K[0,0] = fx
        K_ptr[4] = uniforms.fy;  // K[1,1] = fy
        K_ptr[2] = uniforms.cx;  // K[0,2] = cx
        K_ptr[5] = uniforms.cy;  // K[1,2] = cy
        K_ptr[8] = 1.0f;  // K[2,2] = 1
        result["K"] = K_array;
        
        return result;
    }

    void set_gauss_params(
        py::array_t<float> xyz_ws,
        py::array_t<float> rotations,
        py::array_t<float> scales,
        py::array_t<float> opacities,
        py::array_t<float> sh_coeffs
    ) {
        // Validate input arrays
        auto xyz_buf = xyz_ws.request();
        auto sh_buf = sh_coeffs.request();
        auto rot_buf = rotations.request();
        auto scale_buf = scales.request();
        auto opacity_buf = opacities.request();
        
        if (xyz_buf.ndim != 2 || xyz_buf.shape[1] != 3)
            throw std::runtime_error("xyz_ws must be (N, 3) array");
        if (sh_buf.ndim != 3 || sh_buf.shape[1] != 16 || sh_buf.shape[2] != 3)
            throw std::runtime_error("sh_coeffs must be (N, 16, 3) array");
        if (rot_buf.ndim != 2 || rot_buf.shape[1] != 4)
            throw std::runtime_error("rotations must be (N, 4) array");
        if (scale_buf.ndim != 2 || scale_buf.shape[1] != 3)
            throw std::runtime_error("scales must be (N, 3) array");
        if (opacity_buf.ndim != 2 || opacity_buf.shape[1] != 1)
            throw std::runtime_error("opacities must be (N, 1) array");
        
        buffers.num_splats = xyz_buf.shape[0];

        if ((size_t)sh_buf.shape[0] != buffers.num_splats ||
            (size_t)rot_buf.shape[0] != buffers.num_splats || 
            (size_t)scale_buf.shape[0] != buffers.num_splats ||
            (size_t)opacity_buf.shape[0] != buffers.num_splats
        ) throw std::runtime_error("All input arrays must have the same number of elements N");

        // Prepare inputs
        buffers.xyz_ws.assign((float*)(xyz_buf.ptr), (float*)(xyz_buf.ptr) + xyz_buf.size);
        buffers.rotations.assign((float*)(rot_buf.ptr), (float*)(rot_buf.ptr) + rot_buf.size);
        buffers.assignScalesOpacs(buffers.scales_opacs, buffers.num_splats,
            (float*)(scale_buf.ptr), (float*)(opacity_buf.ptr));
        buffers.sh_coeffs.assign((float*)(sh_buf.ptr), (float*)(sh_buf.ptr) + sh_buf.size);

        buffers.reorderSH(buffers.sh_coeffs);

        if (config.strategy == TrainerConfig::Strategy::MCMC) {
            trainer.resizeDeviceBuffer(buffers.xyz_ws, 3*config.cap_max);
            trainer.resizeDeviceBuffer(buffers.sh_coeffs, 12*4*config.cap_max);
            trainer.resizeDeviceBuffer(buffers.rotations, 4*config.cap_max);
            trainer.resizeDeviceBuffer(buffers.scales_opacs, 4*config.cap_max);
        }
        trainer.copyToDevice(buffers.xyz_ws);
        trainer.copyToDevice(buffers.sh_coeffs);
        trainer.copyToDevice(buffers.rotations);
        trainer.copyToDevice(buffers.scales_opacs);
    }

    void set_pixel_state(
        py::array_t<float> pixel_state,
        py::array_t<int> n_contributors
    ) {
        auto pixel_state_buf = pixel_state.request();
        if (pixel_state_buf.ndim != 3 ||
            pixel_state_buf.shape[0] != uniforms.image_height ||
            pixel_state_buf.shape[1] != uniforms.image_width ||
            pixel_state_buf.shape[2] != 4)
            throw std::runtime_error("pixel_state must be (H, W, 4) array");

        buffers.pixel_state.assign((float*)(pixel_state_buf.ptr), (float*)(pixel_state_buf.ptr) + pixel_state_buf.size);
        trainer.copyToDevice(buffers.pixel_state);

        auto n_contributors_buf = n_contributors.request();
        if (n_contributors_buf.ndim != 3 ||
            n_contributors_buf.shape[0] != uniforms.image_height ||
            n_contributors_buf.shape[1] != uniforms.image_width ||
            n_contributors_buf.shape[2] != 1)
            throw std::runtime_error("n_contributors must be (H, W, 1) array");

        buffers.n_contributors.assign((int32_t*)(n_contributors_buf.ptr), (int32_t*)(n_contributors_buf.ptr) + n_contributors_buf.size);
        trainer.copyToDevice(buffers.n_contributors);
    }

    void set_pixel_state_grad(
        py::array_t<float> v_pixel_state
    ) {
        auto v_pixel_state_buf = v_pixel_state.request();
        if (v_pixel_state_buf.ndim != 3 ||
            v_pixel_state_buf.shape[0] != uniforms.image_height ||
            v_pixel_state_buf.shape[1] != uniforms.image_width ||
            v_pixel_state_buf.shape[2] != 4)
            throw std::runtime_error("v_pixel_state must be (H, W, 4) array");

        buffers.v_pixel_state.assign((float*)(v_pixel_state_buf.ptr), (float*)(v_pixel_state_buf.ptr) + v_pixel_state_buf.size);
        trainer.copyToDevice(buffers.v_pixel_state);
    }

    void projection_forward() {
        uniforms.num_splats = (uint32_t)buffers.num_splats;
        try {
            size_t alloc_reserve = config.strategy == TrainerConfig::Strategy::MCMC ?
                config.cap_max : 0;
            trainer.executeProjectionForward(uniforms, buffers, alloc_reserve);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeProjectionForward failed");
        }
    }
    
    void process_tiles() {
        auto deviceGuard = DeviceGuard(&trainer);

        // calculate index buffer offset (prefix sum)
        try {
            trainer.executeCalculateIndexBufferOffset(buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeCalculateIndexBufferOffset failed");
        }
        if (buffers.num_indices == 0)
            return;

        // generate keys
        try {
            trainer.executeGenerateKeys(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeGenerateKeys failed");
        }

        // sorting
        try {
            trainer.executeSort(uniforms, buffers, -1);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeSort failed");
        }

        // compute tile ranges
        try {
            trainer.executeComputeTileRanges(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeTileRanges failed");
        }

    }
    
    void rasterize_forward() {
        try {
            trainer.executeRasterizeForward(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
        }
    }
    
    void rasterize_backward() {
        try {
            trainer.executeRasterizeBackward(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeBackward failed");
        }
    }

    void forward() {
        auto deviceGuard = DeviceGuard(&trainer);
        projection_forward();
        process_tiles();
        rasterize_forward();
    }

    void backward_optimize(int step) {
        auto deviceGuard = DeviceGuard(&trainer);

        rasterize_backward();

        try {
            trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step+1);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeFusedProjectionBackwardOptimizerStep failed");
        }

    }

    void compute_pixel_loss_grad(size_t train_idx) {
        if (buffers.num_indices == 0)
            return;
        try {
            trainer.executeComputeSSIMGradient(config, uniforms, buffers, train_idx);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeSSIMGradient failed");
        }
    }

    void post_backward_step(int step) {
        if (config.strategy == TrainerConfig::Strategy::Default) {
            try {
                trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
            } catch (const std::runtime_error& err) {
                _THROW_ERROR(std::string(err.what()) + ". trainer.executeDefaultPostBackward failed");
            }
        }
        else if (config.strategy == TrainerConfig::Strategy::MCMC) {
            try {
                trainer.executeMCMCPostBackward(config, uniforms, buffers, step);
            } catch (const std::runtime_error& err) {
                _THROW_ERROR(std::string(err.what()) + ". trainer.executeMCMCPostBackward failed");
            }
        }
    }

    py::dict get_train_image_path(size_t idx) {
        auto image = trainer.get_train_image(idx);
        py::dict result;
        result["image_path"] = image.image_path;
        if (!image.mask_path.empty())
            result["mask_path"] = image.mask_path;
        return result;
    }

    py::dict get_val_image_path(size_t idx) {
        auto image = trainer.get_val_image(idx);
        py::dict result;
        result["image_path"] = image.image_path;
        if (!image.mask_path.empty())
            result["mask_path"] = image.mask_path;
        return result;
    }

    void set_train_image(size_t train_idx) {
        trainer.get_train_camera(train_idx, uniforms);
    }

    void train_step(size_t train_idx, int step) {
        set_train_image(train_idx);
        const int kShDegreeInterval = 1000;
        uniforms.active_sh = std::min(step / kShDegreeInterval, 3);
        uniforms.step = step;

        auto deviceGuard = DeviceGuard(&trainer);

        forward();
        compute_pixel_loss_grad(train_idx);
        backward_optimize(step);

        // post_backward_step(step+1);
        post_backward_step(step);

    }

    void render_train(size_t idx) {
        trainer.get_train_camera(idx, uniforms);
        forward();
    }

    void render_val(size_t idx) {
        trainer.get_val_camera(idx, uniforms);
        forward();
    }

    size_t get_vram_usage() {
        return buffers.getTotalAllocSize();
    }

    size_t get_peak_vram_usage() {
        return trainer.getPeakAllocSize();
    }

    std::map<std::string, std::tuple<size_t, double>> get_timing_breakdown() {
        return PerfTimer::get_summary();
    }

    std::map<std::string, size_t> get_vram_breakdown() {
        return buffers.getVramBreakdown();
    }

    void write_ply(std::string filename) {
        trainer.writePLY(filename, buffers);
    }

    void cleanup() {
        trainer.cleanupBuffers(buffers);
        trainer.cleanup();
    }
};

#define DEF_BUFFER_0(key, buffer_name, dtype, ...) \
    def_property_readonly(#key, [] (VkSplat &self) { \
        return self.buffer_to_array<dtype, dtype>(self.buffers.buffer_name, __VA_ARGS__); \
    })

#define DEF_BUFFER(key, dtype, ...) \
    DEF_BUFFER_0(key, key, dtype, __VA_ARGS__)

#define DEF_ARRAY(key, dtype) \
    def("_set_"#key, [] (VkSplat &self, py::array_t<dtype> array) { \
        self.array_to_buffer(array, self.buffers.key); \
    }, "Directly set `"#key"` buffer (use at own risk)", py::arg(#key))

#define DEF_BUFFER_ARRAY(key, dtype, ...) \
    DEF_BUFFER(key, dtype, __VA_ARGS__) \
    .DEF_ARRAY(key, dtype)

PYBIND11_MODULE(vksplat, m) {
    m.doc() = "Vulkan Gaussian Rasterization Python Bindings";
    
    py::class_<VkSplat>(m, "VkSplat")
        .def(py::init<>())
        .def("initialize", &VkSplat::initialize, "Initialize the trainer with SPIRV code")
        .def("set_train_config", &VkSplat::set_train_config, "Set training configuration")
        .def_property_readonly("num_train", &VkSplat::num_train)
        .def_property_readonly("num_val", &VkSplat::num_val)
        .def("get_train_image_path", &VkSplat::get_train_image_path)
        .def("get_val_image_path", &VkSplat::get_val_image_path)
        .def("get_train_image", &VkSplat::get_train_image)
        .def("get_val_image", &VkSplat::get_val_image)
        .def("set_uniforms", &VkSplat::set_uniforms, "Set trainer uniforms (image/camera info)",
             py::arg("active_sh"), py::arg("world_view_transform"),
             py::arg("image_height"), py::arg("image_width"),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("is_fisheye"))
        .def("get_uniforms", &VkSplat::get_uniforms, "Get trainer uniforms as a dict")
        .def("set_gauss_params", &VkSplat::set_gauss_params, "Set Gaussian parameters",
             py::arg("xyz_ws"), py::arg("sh_coeffs"), py::arg("rotations"), py::arg("scales"), 
             py::arg("opacities"))
        .def("set_pixel_state", &VkSplat::set_pixel_state, "Set output pixels and n_contributors",
             py::arg("pixel_state"), py::arg("n_contributors"))
        .def("set_pixel_state_grad", &VkSplat::set_pixel_state_grad, "Set gradient of output pixels",
             py::arg("v_pixel_state"))
        .def("projection_forward", &VkSplat::projection_forward,
             "Execute the forward projection trainer")
        .def("process_tiles", &VkSplat::process_tiles,
             "Process tiles")
        .def("rasterize_forward", &VkSplat::rasterize_forward,
             "Execute the forward rasterization trainer")
        .def("rasterize_backward", &VkSplat::rasterize_backward,
             "Execute the backward rasterization trainer")
        .def("forward", &VkSplat::forward, "Execute forward all the way")
        .def("compute_pixel_loss_grad", &VkSplat::compute_pixel_loss_grad, "Execute compute gradient of pixel loss")
        .def("set_train_image", &VkSplat::set_train_image, "Train on the give image for one step.")
        .def("train_step", &VkSplat::train_step, "Train on the give image for one step.")
        .def("render_train", &VkSplat::render_train, "Render the train camera pose to buffer.")
        .def("render_val", &VkSplat::render_val, "Render the eval camera pose to buffer.")
        .def("post_backward_step", &VkSplat::post_backward_step,
             "Run this after each step for densification.")
        .def("get_vram_usage", &VkSplat::get_vram_usage, "Get VRAM Usage")
        .def("get_peak_vram_usage", &VkSplat::get_peak_vram_usage, "Get peak VRAM Usage")
        .def("get_timing_breakdown", &VkSplat::get_timing_breakdown, "Get timing for each step since start")
        .def("get_vram_breakdown", &VkSplat::get_vram_breakdown, "Get VRAM for each buffer at the current moment")
        .def("write_ply", &VkSplat::write_ply, "Save PLY file")
        .def("cleanup", &VkSplat::cleanup, "Cleanup Vulkan resources")
        .DEF_BUFFER_ARRAY(xyz_ws, float, 3, (size_t)-1, "mean")
        .DEF_BUFFER_ARRAY(sh_coeffs, float, 16, 3, "sh")
        .DEF_BUFFER_ARRAY(rotations, float, 4)
        .DEF_BUFFER_0(scales, scales_opacs, float, 4, (size_t)-1, "", 3, 0)
        .DEF_BUFFER_0(opacities, scales_opacs, float, 4, (size_t)-1, "", 1, 3)
        .DEF_BUFFER_ARRAY(tiles_touched, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(rect_tile_space, int64_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(radii, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(depths, float, 1)
        .DEF_BUFFER_ARRAY(inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(rgb, float, 3)
        .DEF_BUFFER_ARRAY(index_buffer_offset, int32_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_keys_1, sortingKey_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_gauss_idx_1, int32_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_keys_2, sortingKey_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_gauss_idx_2, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(tile_ranges, int32_t, 2)
        .DEF_BUFFER_ARRAY(pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(n_contributors, int32_t, self.uniforms.image_width, 1)
        .DEF_BUFFER_ARRAY(ssim_map, float, self.uniforms.image_width, 12)
        .DEF_BUFFER_ARRAY(v_pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(v_xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(v_depths, float, 1)
        .DEF_BUFFER_ARRAY(v_inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(v_rgb, float, 3)
        .def("_set_num_splats", [] (VkSplat &self, size_t n) {
            self.buffers.num_splats = n;
        }, "Directly set buffers.num_splats (use at own risk)", py::arg("num_splats"))
        ;

    py::register_exception<std::runtime_error>(m, "RuntimeError");
}
