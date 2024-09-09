from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

# Path to the .pyx file
pyx_file = os.path.join("model_chromatogram", "utils", "autocorr_data.pyx")

# Define the setup
setup(
    ext_modules=cythonize(pyx_file, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
