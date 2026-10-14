"""Build script for the C++/Cython extension ``autokinetics._ext._core``.

All C++ sources below ``cpp/src`` are compiled into a single extension module so
that kinetics objects (reactor systems) can be handed directly to the numerics
layer (integrators, linear algebra) without crossing module boundaries.
"""

import os
from glob import glob

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = os.path.dirname(os.path.abspath(__file__))

extra_compile_args = ["-std=c++17", "-O3", "-Wall", "-Wno-unused-function"]
if os.environ.get("AUTOKINETICS_NATIVE", "0") == "1":
    extra_compile_args.append("-march=native")
if os.environ.get("AUTOKINETICS_DEBUG", "0") == "1":
    extra_compile_args = ["-std=c++17", "-O0", "-g", "-Wall", "-fsanitize=address,undefined"]

cpp_sources = sorted(glob(os.path.join("cpp", "src", "**", "*.cpp"), recursive=True))

ext = Extension(
    "autokinetics._ext._core",
    sources=[os.path.join("src", "autokinetics", "_ext", "_core.pyx"), *cpp_sources],
    include_dirs=[os.path.join(HERE, "cpp", "include"), np.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=["-fsanitize=address,undefined"] if os.environ.get("AUTOKINETICS_DEBUG") == "1" else [],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
        include_path=[os.path.join("src", "autokinetics", "_ext")],
    ),
)
