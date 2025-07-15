"""
This script reads in the ECCO2 state estimate outputs from the APDRC server,
subsets them to a specific region and time duration and crops (if necessary)

It then saves the processed datasets to NetCDF files.

Defaults to the Eastern Tropical Pacific region (5N to 23.75N, 245E to 267.5E)
and the time period from 2004 to 2022.
This region and time period can be adjusted by changing the lat/lon ranges
and time range in the script.

AUTHOR: Parth Sastry
"""

# NOTE - WORK IN PROGRESS. DOES NOT WORK YET.

import xarray as xr
import numpy as np
import argparse
from helperLibrary import crop


def processDataset(
        datasetURL: str,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        time_range: tuple[np.datetime64, np.datetime64],
        crop_lat_range: tuple[float, float],
        crop_lon_range: tuple[float, float]
):
    # Open the dataset
    with xr.open_dataset(datasetURL, engine='pydap') as ds:
        # Subset the dataset
        ds = ds.sel(
            lat=slice(*lat_range),
            lon=slice(*lon_range),
            time=slice(*time_range),
            drop=True
        )

        # Crop the dataset if necessary
        if crop_lat_range or crop_lon_range:
            ds = crop(ds, lat_range=crop_lat_range, lon_range=crop_lon_range)

    return ds


def combineECCO2Datasets(
        urls: list[str],
        lat_range: tuple[float, float] = (5, 23.75),
        lon_range: tuple[float, float] = (245, 267.5),
        time_range: tuple[np.datetime64, np.datetime64] = (
            np.datetime64('2004-01-01'), np.datetime64('2022-12-31')
        ),
        crop_lat_range: tuple[float, float] = (18, 25),
        crop_lon_range: tuple[float, float] = (240, 270)
):
    # Combine the datasets
    for url in urls:
        ds = processDataset(
            datasetURL=url,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            crop_lat_range=crop_lat_range,
            crop_lon_range=crop_lon_range
        )
        if 'combined' not in locals():
            combined = ds
        else:
            combined = xr.merge([combined, ds])

    return combined


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process ECCO2 state estimate datasets from APDRC.'
    )
