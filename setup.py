"""
setup.py
Build configuration for SMR-Engine Cooperative Warp-Compaction WMMA Extension.
Compiles kernel.cu into an optimized PyTorch C++/CUDA hardware module.
"""

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_89,code=sm_89',
        '-gencode=arch=compute_90,code=sm_90',
        '-gencode=arch=compute_120,code=sm_120',
    ]
}

if torch.cuda.is_available():
    ext_modules = [
        CUDAExtension(
            name='smr_cuda',
            sources=['kernel.cu'],
            extra_compile_args=extra_compile_args
        )
    ]
else:
    ext_modules = []

setup(
    name='smr_engine_cuda',
    version='1.0.0',
    description='SMR-Engine Cooperative Warp-Compaction WMMA CUDA Kernel',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {}
)
