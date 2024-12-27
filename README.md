# Capstone_BergomiVIX

This repo contains the research for our Capstone project on VIX Futures and rough volatility. It contains a few folders, `.ipynb` files with the actual research and some `.py` files with useful functions.

## Python files

1. `DataLoader.py`:

It contains a class `DataLoader` to streamline reading and dumping data from within the folder. It is based on `polars` for data manipulation and supports `.csv` and `.parquet` extensions.

2. `Calibration.py`:

This file contains functions related to the calibration task, such as implementation of the Carr-Madran formula for power strips of the VIX, the Bayer-Friz-Gatheral approximation of the lognormal variance of VIX and SVI fitting.

3. `utils.py`:

General utility functions, especially to streamline computations and constants related to the Hurst exponent.

## Notebooks

1. `acf_estimator.ipynb`:

This notebook computes and analyzes the Rendleman-Bartter estimator of realized variance. It highlights its roughness and compares the ACF-based estimate of $H$ with the rough Bergomi, VIX futures-based one.

2. `vix_futuri_cioccolats_diffevo.ipynb` and `vix_futuri_cioccolats_svi.ipynb`:

Core of the research - runs the calibration over the whole sample and examines the output. The `_svi` version tries to look at including SVI on some days.

3. `are_you_reruira.ipynb`:

:warning: @Nicco add something here for the trading strategy notebook

4. `macarons_days.ipynb`:

Explores the days in which the calibration to the forward variance curve failed and fixes some of the issues. Was not needed in the end as we calibrated to VIX futures instead.

5. `rezdora_calibration.ipynb`:

More detail analysis of the inital calibration and comparison with differential evolution calibration.

## Folders

1. `docs`:

Contains the two main papers we followed and some lecture notes by Jim, as well as our final dissertation.

2. `OldResearch`:

Contains notebooks with some of the research that ended up being not too relevant. They are not well-maintained but may contain somewhat relevant information here and there.
- `gabo_research.ipynb` is a first attempt to replicate Jacquier et al., which we later put aside to focus more on Bayer et al.
- `calibrate_cream_cheese.ipynb` contains the first calibration we tried (to the forward variance curve rather than VIX term structure) and compares L-BFGS-B with differential evolution.

3. `params`:

Contains parquets with estimated parameters over different date ranges and with different estimation techniques. The final version which we included in our paper is in `params_futures_diffevo_20060224_20230831.parquet`.

4. `Plots`:

Contains the plots worht saving and those that we included in the report.

