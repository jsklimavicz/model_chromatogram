# model_chromatogram

For very mathematical chromatograms modelling.

For more inforamtion on the workings of the program and calculations, see `./docs/Chromatogram Model.pdf`.

## Setup

### Poetry Install

A `pyproject.toml` is included; to install dependences, run

```bash
poetry install
```

### Cython Modules

Then, to create the appropriate cython modules, run

```bash
python setup.py build_ext --inplace
```

### Julia Modules

Lastly, the pressure module can use julialang to calculate viscosities and pressures for the mobile phase. This requires Julia to be installed; see [here](https://julialang.org/downloads/) for installations.

The file `./model_chromatogram/user_parameters.py` mus be updated to contain the correct path for the julia executable. The `JULIA_PARAMTERS` is a dictionary, and the key `julia` is to be used with the string path of the executable, e.g.

```python
JULIA_PARAMTERS = {
    # Path to your system's Julia executable.
    "julia": "/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia",
}
```

The executable path should have the dependencies `"CSV", "DataFrames", "LsqFit"` installed.

This can be done in julia with

```julia
using Pkg

Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("LsqFit")
```

**NOTE**: The first time the code runs using julia, expect some level of setup as `juliacall` compiles some binaries. Later, the calls should be significantly faster.


#### Don't want to use julia?

If you do not wish to use julia, set the julia path to `None`:

```python
JULIA_PARAMTERS = {
    # Path to your system's Julia executable.
    "julia":None,
}
```

A `scipy` interpolation will be performed to calculate viscosities instead. However, this method is *much* slower, taking about 50x longer on Apple Silicon M1 and M3 chip sets.
