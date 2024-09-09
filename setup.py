from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

files = [
    "autocorr_data.pyx",
    "exponnorm_functions.pyx",
]

pyx_files = []
for file in files:
    pyx_files.append(os.path.join("model_chromatogram", "utils", file))

# Define the setup
setup(
    ext_modules=cythonize(pyx_files, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
