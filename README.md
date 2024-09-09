# model_chromatogram
For very mathematical chromatograms modelling. 

For more inforamtion on the workings of the program and calculations, see `./docs/Chromatogram Model.pdf`. 

# Setup

A `pyproject.toml` is included; to install dependences, run 
```
poetry install
```

Then, to create the appropriate cython modules, run 
```
python setup.py build_ext --inplace
```