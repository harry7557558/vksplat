import os
import sys
import platform
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, Extension

__version__ = "0.1.0"

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "vksplat",
        [
            "src/buffer.cpp",
            "src/colmap_reader.cpp", 
            "src/knn.cpp",
            "src/scheduler.cpp",
            "src/gs_pipeline.cpp",
            "src/gs_renderer.cpp",
            "src/gs_trainer.cpp",
            "src/perf_timer.cpp",
            "src/python_bindings.cpp",
        ],
        include_dirs=[
            "src/",
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
    ),
]

def find_vulkan():
    """Find Vulkan SDK installation"""
    # Check environment variable first
    vulkan_sdk = os.environ.get('VULKAN_SDK')
    if vulkan_sdk and os.path.exists(vulkan_sdk):
        return vulkan_sdk
    
    # Platform-specific default locations
    if platform.system() == "Windows":
        # Windows default installation paths
        possible_paths = [
            "C:/VulkanSDK/*/",
            os.path.expanduser("~/VulkanSDK/*/")
        ]
    elif platform.system() == "Darwin":  # macOS
        possible_paths = [
            "/usr/local/vulkan/macOS/",
            "~/VulkanSDK/macOS/"
        ]
    else:  # Linux
        possible_paths = [
            "/usr/include/vulkan/",
            "/usr/local/include/vulkan/",
            "~/VulkanSDK/*/x86_64/"
        ]
    
    import glob
    for path_pattern in possible_paths:
        expanded = os.path.expanduser(path_pattern)
        matches = glob.glob(expanded)
        if matches:
            return matches[0]
    
    return None

def configure_build():
    """Configure build settings based on platform and available libraries"""

    # Find Vulkan
    vulkan_path = find_vulkan()
    if not vulkan_path:
        raise RuntimeError("Vulkan SDK not found. Please install Vulkan SDK and set VULKAN_SDK environment variable.")
    print(f"Found Vulkan SDK at: {vulkan_path}")
    
    # Configure extension
    ext = ext_modules[0]
    
    # Add Vulkan include and library paths
    if platform.system() == "Windows":
        ext.include_dirs.extend([
            os.path.join(vulkan_path, "Include"),
        ])
        ext.library_dirs = [os.path.join(vulkan_path, "Lib")]
        ext.libraries = ["vulkan-1"]
        ext.define_macros = [("VK_USE_PLATFORM_WIN32_KHR", None)]
    elif platform.system() == "Darwin":  # macOS
        ext.include_dirs.extend([
            os.path.join(vulkan_path, "include"),
        ])
        ext.library_dirs = [os.path.join(vulkan_path, "lib")]
        ext.libraries = ["vulkan"]
        ext.define_macros = [("VK_USE_PLATFORM_MACOS_MVK", None)]
    else:  # Linux
        ext.include_dirs.extend([
            os.path.join(vulkan_path, "include") if "/include/vulkan" not in vulkan_path else vulkan_path.replace("/vulkan", ""),
            # "/usr/include",
            # "/usr/local/include"
        ])
        ext.library_dirs = [
            os.path.join(vulkan_path, "lib") if vulkan_path else "/usr/lib",
            # "/usr/local/lib"
        ]
        ext.libraries = ["vulkan"]
        ext.define_macros = [("VK_USE_PLATFORM_XLIB_KHR", None)]
    
    # Add GLM (header-only library)
    # Try to find system GLM first
    glm_paths = [
        "/usr/include/glm",
        "/usr/local/include/glm",
        "/opt/homebrew/include/glm",  # macOS Homebrew
        "third_party/glm/glm"  # Local copy
    ]
    
    glm_found = False
    for glm_path in glm_paths:
        if os.path.exists(os.path.join(glm_path, "glm.hpp")):
            ext.include_dirs.append(os.path.dirname(glm_path))
            glm_found = True
            print(f"Found GLM at: {glm_path}")
            break
    
    if not glm_found:
        print("GLM not found in system paths. Cloning from GitHub...")
        glm_dir = "third_party/glm"
        
        # Create third_party directory if it doesn't exist
        os.makedirs("third_party", exist_ok=True)
        
        # Clone GLM if directory doesn't exist
        if not os.path.exists(glm_dir):
            try:
                subprocess.run([
                    "git", "clone", "--depth", "1", "--branch", "0.9.9.8",
                    "https://github.com/g-truc/glm.git", glm_dir, "--depth", "1"
                ], check=True, capture_output=True, text=True)
                print(f"Successfully cloned GLM to {glm_dir}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone GLM: {e.stderr}")
            except FileNotFoundError:
                raise RuntimeError("Git not found. Please install git or manually place GLM in third_party/glm/")
        
        # Add GLM include path
        ext.include_dirs.append(glm_dir)
    
    # Compiler flags
    if platform.system() != "Windows":
        ext.extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-O3",  # Release optimization
            "-DVERSION_INFO=\"{}\"".format(__version__)
        ]
    else:
        ext.extra_compile_args = [
            "/O2",  # Release optimization  
            "/std:c++17",
            "/DVERSION_INFO=\"{}\"".format(__version__)
        ]
    cflags = os.getenv("CFLAGS")
    if cflags is not None:
        ext.extra_compile_args += cflags.strip().split()

    # Set visibility and optimization
    if platform.system() != "Windows":
        ext.extra_compile_args.extend([
            "-fvisibility=hidden"
        ])


class CustomBuildExt(build_ext):
    """Custom build extension to handle dependencies"""
    
    def build_extensions(self):
        configure_build()
        super().build_extensions()


setup(
    name="vksplat",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "opencv-python",
        "tqdm",
        "torchmetrics[image]>=1.0.1"  # TODO: possibly get a Vulkan version
    ],
    setup_requires=[
        "pybind11>=2.11.1",
        "setuptools",
    ],
)
