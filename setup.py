import os
import tempfile
import subprocess
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import distutils.sysconfig
from distutils import ccompiler


def has_flag(compiler, flagname):
    """Return True if the compiler supports a given flag."""
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write("int main(void) { return 0; }")
        fname = f.name
    try:
        cmd = compiler.compiler_so + [flagname, fname]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except Exception:
        return False
    finally:
        os.remove(fname)
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] flag. Assume that the compiler supports at least C++11."""
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


def get_openmp_flags(compiler):
    """Return a tuple of (extra_compile_args, extra_link_args) if OpenMP is supported, or empty lists otherwise."""
    omp_compile_flag = "-fopenmp"
    omp_link_flag = "-fopenmp"
    if has_flag(compiler, omp_compile_flag):
        return [omp_compile_flag], [omp_link_flag]
    else:
        return [], []


# Build a list of Extension objects.
files = [
    "utils.pyx",
    "autocorr_data.pyx",
    "exponnorm_functions.pyx",
    "viscosity.pyx",
    "pressure.pyx",
    "baseline.pyx",
    "pentadiagonal.pyx",
    "signal_smoothing.pyx",
    "find_peaks.pyx",
    "compound_calculations.pyx",
]

extensions = []
for file in files:
    # Get the full path to the .pyx file.
    pyx_path = os.path.join("model_chromatogram", "utils", file)
    # Derive a module name from the file path (replace os.sep with '.' and strip the extension).
    module_name = os.path.splitext(pyx_path)[0].replace(os.sep, ".")
    ext = Extension(
        module_name,
        [pyx_path],
        include_dirs=[numpy.get_include()],
    )
    extensions.append(ext)

# Use a temporary compiler instance to detect flags.
compiler = ccompiler.new_compiler()

# customize the compiler if needed (e.g. for platform specifics)
distutils.sysconfig.customize_compiler(compiler)

omp_compile_args, omp_link_args = get_openmp_flags(compiler)

# Optionally, you could also set high optimization flags.
common_compile_args = ["-O3"]
for ext in extensions:
    ext.extra_compile_args = common_compile_args + omp_compile_args
    ext.extra_link_args = omp_link_args

setup(
    name="model_chromatogram",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
