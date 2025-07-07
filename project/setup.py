import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name="project",
    ext_modules=cythonize("project/project.pyx", language_level="3"),
    include_dirs=[numpy.get_include()]
)
