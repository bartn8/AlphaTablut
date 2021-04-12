from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    # Everything but primes.pyx is included here.
    Extension("*", ["*.pyx"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    include_dirs = [numpy.get_include()])
]

setup(
    name="AlphaTablut",
    package_dir={'AlphaTablut': ''},
    ext_modules=cythonize(extensions),
)