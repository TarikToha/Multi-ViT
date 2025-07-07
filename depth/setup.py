import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    name="depth_project",
    ext_modules=cythonize("depth/depth_project.pyx", language_level="3"),
    include_dirs=[np.get_include()]
)
