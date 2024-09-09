from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

pyx_files = [
    os.path.join("model_chromatogram", "utils", "autocorr_data.pyx"),
    os.path.join("model_chromatogram", "utils", "exponnorm_functions.pyx"),
]

# Define the setup
setup(
    ext_modules=cythonize(pyx_files, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
