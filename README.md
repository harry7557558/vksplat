# VkSplat

[![Website](https://img.shields.io/website?url=https://harry7557558.github.io/vksplat/&logo=github)](https://harry7557558.github.io/vksplat/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.00219-b31b1b.svg)](https://arxiv.org/abs/2605.00219)
![License](https://img.shields.io/github/license/harry7557558/vksplat)

This project provides functionality for training 3D Gaussian Splatting (3DGS) models, using Vulkan compute backend with Python binding.

This is code for paper "VkSplat: High-Performance 3DGS Training in Vulkan Compute".

Features:
- Cross vendor 3DGS training (tested with NVIDIA, AMD, Intel®)
- High performance (over 3.3x faster training compared to GSplat)
- Memory efficiency (9 million SH3 Gaussians in 8GB VRAM for garden scene with MCMC)
- Quality matching baseline (identical PSNR, SSIM, LPIPS compared to GSplat)
- Default (original ADC from Inria) and MCMC densification
- Support for non-centered and distorted/fisheye cameras


## Prerequisites

### System Requirements
- Vulkan SDK installed
- Python 3.7+
- Python3 dependencies for building pybind11 C++ extension AND/OR CMake 3.16+ (see "Installation" section below)
- C++17 compatible compiler

#### Tested with
- Vulkan 1.3 and 1.4
- Windows 10/11, Ubuntu 22.04/24.04/25.04
- NVIDIA RTX 3090, NVIDIA RTX 4080 Super, NVIDIA RTX 5070 Laptop, AMD Radeon RX 7800 XT, Intel® UHD Graphics 750, Intel® UHD Graphics 770

We also received feedback from users who successfully ran VkSplat on Mac devices using MoltenVK.

## Installation

### Method 1: Using pip

This is the recommended option if you are trying the method and want to see it working quickly/reliably.
This assumes you already have necessary dependencies to build a Pybind11 extension (setuptools, pybind11, etc.).

```bash
cd /path/to/vksplat
pip install -e . --no-build-isolation  # optionally with -v
```

Import from Python:

```py
import vksplat
```

This should work anywhere in your Python environment. Be careful with folders with the same name as `vksplat` as they may take precedence during imports.

### Method 2: Using CMake

This is the recommended option for development.

Linux:

```bash
cd /path/to/vksplat
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..  # or Debug
make -j
```

Windows:

```bash
cd /path/to/vksplat
cmake -B build
cmake --build build --config Release  # or Debug
```

Import from Python (from `vksplat` folder):
```py
from build import vksplat  # or build.Debug, build.Release for MSVC
```

Make sure relevant Python dependencies (`numpy`, `opencv-python`, `tqdm`) are installed when running `simple_trainer.py`.
Optionally, install `torchmetrics[image]>=1.0.1` if you want to run evaluation.

Be careful if you have a package with the same name as `build`, as it may take precedence during imports.

## Quick Start

See `simple_trainer.py` for an example. It trains a 3DGS model using Vulkan, saves results to file, prints time and VRAM breakdown, and computes evaluation metrics using torchmetrics library. A CUDA-compatible GPU is not required for evaluation. It also provides a function to run benchmark across Mip-NeRF 360 dataset.

Before running `simple_trainer.py`, make the following edits if needed:
- Near the beginning of the file, set `TRAIN_DEVICE` to the device you want to use
- In `train` function, choose the way your import `vksplat` based on how you installed it
- Adjust parameters in `TrainerConfig` and `MCMCTrainerConfig` classes, particularly `output_dir`, `dataset_dir`, `image_dir`, and `cap_max` if you are using MCMC
- Inside the `__name__ == "__main__"` block at the end of file, choose whether you want to train default, train MCMC, or run batch evaluation.
- If you are running batch evaluation, adjust code in `benchmark_mipnerf360` function if needed.
- If you want to use an in-browser viewer (similar to the one used by GSplat) during training, set `enable_viewer` in trainer config to `True`.

Running the code should create a work folder. After training, you may find training time and memory in `train.json`, metrics in `eval.json`, saved PLY file in `splat.ply`, as well as validation renders.

If you see message similar to "Shaders must be compiled with USE_XXX=1" for the device you use for training, adjust `vksplat/slang/config.slang`, particularly `USE_EMULATED_INT64` and `USE_EMULATED_F32_ATOMIC` macros. You must recompile shaders for this edit to take effect (see "Recompile shaders" section below).


## Development

Slang code in `vksplat/slang/`.

Compiled shaders in `vksplat/shader/generated`:
- SPIR-V binaries (`.spv`) are loaded by program at run time (see `simple_trainer.py`)
- Tested with `slang-2026.2.1-linux-x86_64`, other versions may also work

Vulkan/C++ code in `vksplat/src/`:
- `buffer`: Refactored buffers relevant for 3DGS training
- `gs_pipeline`: Vulkan abstraction
- `gs_renderer`: Rendering functionality, inherited from `gs_pipeline`
- `gs_trainer`: Training functionality, inherited from `gs_renderer`


### Recompile shaders

To recompile shaders after update, run `python3 compile_shaders.py` from project root directory.

To force recompile all shaders without caching, use `python3 compile_shaders.py --force`.

If you add new shader source files, you must update `compile_shaders.py`.


### Recompile Vulkan/C++ code

#### For pip:
```bash
cd /path/to/vksplat
python -m pip install -e . --no-build-isolation -v
```
In some cases, you may need to delete the `build` folder before running the command to clear the cache.

#### For CMake:
```bash
cd /path/to/vksplat
make -j
```

If you add new source files, you must add them to the list of sources in `setup.py` and `CMakeLists.txt` before running the recompilation commands.



## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{chen2026vksplat,
  booktitle = {Eurographics 2026 - Short Papers},
  title     = {{VkSplat: High-Performance 3DGS Training in Vulkan Compute}},
  author    = {Chen, Jingxiang and Ibrahim, Mohamed and Liu, Yang},
  year      = {2026},
  publisher = {The Eurographics Association},
  ISSN      = {2309-5059},
  ISBN      = {978-3-03868-299-8},
  DOI       = {10.2312/egs.20261024}
}
```
