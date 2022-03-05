# Detecting temperature targets
***
Neural networks are trained on CMIP6 data to detect the remaining number of years until specific temperature targets are reached.

## Tensorflow Code
***
This code was written in python 3.9.7, tensorflow 2.7.0, tensorflow-probability 0.15.0 and numpy 1.21.4. 

## General Notes
***

### Python Environment
The following python environment was used to implement this code.
```
conda create --name env-noah python=3.9
conda activate env-noah
pip install tensorflow==2.7.0
pip install tensorflow-probability==0.15.0
pip install --upgrade numpy scipy pandas statsmodels matplotlib seaborn palettable progressbar2 tabulate icecream flake8 keras-tuner sklearn jupyterlab black isort jupyterlab_code_formatter
pip install -U scikit-learn
pip install silence-tensorflow tqdm
conda install -c conda-forge cmocean cartopy
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda install -c conda-forge nc-time-axis
```

### Credits
This work is a collaborative effort between[Dr. Noah Diffenbaugh](https://earth.stanford.edu/people/noah-diffenbaugh#gs.runods) and  [Dr. Elizabeth A. Barnes](https://barnes.atmos.colostate.edu). 

#### Funding sources

### References
[1] None.

### License
This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)




