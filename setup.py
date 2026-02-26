"""
Metal-GS build script.

Compiles Metal shaders (.metal) → .metallib AOT (Ahead-of-Time) using Xcode toolchain.
Then compiles ObjC++ (.mm) + C++ (.cpp) with Metal framework linkage.

Usage:
  pip install -e .          # editable install (recommended for dev)
  python setup.py build_ext --inplace   # build in-place
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"
KERNEL_DIR = CSRC / "kernels"

# Precision control: ENABLE_BF16=0 for M1 (FP32 fallback), =1 for M4+ (BF16 training)
METAL_DEFINES = {
    "ENABLE_BF16": "1",  # Set to "1" when targeting M4/Apple GPU Family 9+
}


class MetalBuildExt(build_ext):
    """Custom build_ext that:
    1. AOT-compiles all .metal shaders to a single metal_gs.metallib
    2. Compiles C++/ObjC++ extensions with Metal framework links
    
    Forces Apple clang (GCC cannot compile ObjC++/Metal).
    """

    def build_extensions(self):
        # ---- Step 1: AOT compile Metal shaders ----
        self._compile_metal_shaders()
        
        # ---- Step 2: Force Apple clang ----
        self.compiler.set_executable('compiler_so', 'clang++')
        self.compiler.set_executable('compiler_cxx', 'clang++')
        self.compiler.set_executable('compiler', 'clang')
        self.compiler.set_executable('linker_so', 'clang++')

        # Tell the compiler about .mm files
        self.compiler.src_extensions.append('.mm')

        # Patch compilation for ObjC++
        original_compile = self.compiler._compile

        def patched_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            _postargs = list(extra_postargs)
            if src.endswith('.mm'):
                _postargs = [a for a in _postargs if a not in ('-std=c++17',)]
                _postargs += ['-ObjC++', '-std=c++17', '-fobjc-arc']
            return original_compile(obj, src, ext, cc_args, _postargs, pp_opts)

        self.compiler._compile = patched_compile
        super().build_extensions()

    def _compile_metal_shaders(self):
        """AOT compile all .metal files in csrc/kernels/ to metal_gs.metallib."""
        metal_srcs = sorted(KERNEL_DIR.glob("*.metal"))
        if not metal_srcs:
            raise RuntimeError(f"No .metal files found in {KERNEL_DIR}")

        print("\n" + "="*72)
        print("  Metal-GS: AOT Metal Shader Compilation")
        print("="*72)

        # Prepare defines for Metal compiler
        metal_flags = []
        for key, val in METAL_DEFINES.items():
            metal_flags.extend(["-D", f"{key}={val}"])

        # Determine MSL version: metal3.2 required for bfloat (M4+)
        msl_version = "metal3.2" if METAL_DEFINES.get("ENABLE_BF16") == "1" else "metal3.0"

        air_files = []
        for metal_file in metal_srcs:
            air_file = metal_file.with_suffix(".air")
            print(f"  [metal] {metal_file.name} → {air_file.name}  (MSL {msl_version})")
            cmd = [
                "xcrun", "-sdk", "macosx", "metal",
                "-c", str(metal_file),
                "-o", str(air_file),
                "-ffast-math",
                "-std=" + msl_version,
            ] + metal_flags
            
            try:
                subprocess.check_call(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] Metal compilation failed for {metal_file.name}")
                raise
            air_files.append(air_file)

        # Link .air → .metallib
        metallib_path = KERNEL_DIR / "metal_gs.metallib"
        print(f"\n  [metallib] Linking {len(air_files)} shaders → {metallib_path.name}")
        cmd = [
            "xcrun", "-sdk", "macosx", "metallib",
            *[str(f) for f in air_files],
            "-o", str(metallib_path),
        ]
        
        try:
            subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] metallib linking failed")
            raise

        # Clean up .air files
        for f in air_files:
            f.unlink(missing_ok=True)

        print(f"  [✓] {metallib_path.relative_to(ROOT)}")
        print("="*72 + "\n")


def get_pybind11_include():
    """Get pybind11 include path."""
    import pybind11
    return pybind11.get_include()


def get_torch_include_dirs():
    """Get PyTorch C++ include paths for torch::Tensor API."""
    import torch.utils.cpp_extension
    return torch.utils.cpp_extension.include_paths()


def get_torch_library_dirs():
    """Get PyTorch library paths for libtorch linking."""
    import torch.utils.cpp_extension
    return torch.utils.cpp_extension.library_paths()


def build_extension():
    """Build the main C++ extension."""
    # Extra compile args for ObjC++ / C++17
    extra_compile_args = [
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wno-unused-variable",
        "-Wno-unused-function",
        # ObjC ARC
        "-fobjc-arc",
    ]

    extra_link_args = [
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "MetalPerformanceShaders",
    ]

    ext = Extension(
        name="metal_gs._metal_gs_core",
        sources=[
            "csrc/bindings.cpp",
            "csrc/metal_wrapper.mm",
        ],
        include_dirs=[
            "csrc",
            get_pybind11_include(),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    return ext


def build_uma_extension():
    """Build the CPU-GPU Zero-Copy Synergy extension.
    
    This module uses torch::Tensor C++ API + ATen MPS internals to directly
    access MPS tensor data via [MTLBuffer contents] — true UMA zero-copy.
    Links against libtorch_cpu, libtorch_python, and libc10.
    """
    torch_inc = get_torch_include_dirs()
    torch_lib = get_torch_library_dirs()

    extra_compile_args = [
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-fobjc-arc",
        # PyTorch build flags
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ]

    extra_link_args = [
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "MetalPerformanceShaders",
        # Link PyTorch libraries
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_python",
        "-lc10",
    ]
    # Add rpath for torch libs
    for ld in torch_lib:
        extra_link_args.extend(["-L" + ld, "-Wl,-rpath," + ld])

    ext = Extension(
        name="metal_gs._metal_gs_uma",
        sources=[
            "csrc/cpu_gpu_synergy.mm",
        ],
        include_dirs=[
            "csrc",
            get_pybind11_include(),
        ] + torch_inc,
        library_dirs=torch_lib,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    return ext


setup(
    name="metal-gs",
    version="0.2.0",
    description="Apple Silicon-native Gaussian Splatting operators",
    packages=find_packages(),
    ext_modules=[build_extension(), build_uma_extension()],
    cmdclass={"build_ext": MetalBuildExt},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pybind11>=2.11",
    ],
    # Include Metal shader sources in the package (compiled at runtime)
    package_data={
        "metal_gs": ["*.metal"],
    },
)
