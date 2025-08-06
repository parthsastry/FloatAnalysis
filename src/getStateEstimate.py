"""
This script reads in the ECCO2 state estimate outputs,
subsets them to a specific region and crops (if necessary).
It then computes the associated density grid and maps UVEL and VVEL
to the density grid.

It then saves the processed datasets to NetCDF files.

Defaults to the Eastern Tropical Pacific region (5N to 23.75N, 245E to 267.5E)

This region can be adjusted by changing the lat/lon ranges in the script.

AUTHOR: Parth Sastry
"""

# NOTE - WORK IN PROGRESS. DOES NOT WORK YET.

import xarray as xr
import numpy as np
import argparse
from helperLibrary import crop


def processDataset(
        datasetLocation: str,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        crop_lat_range: tuple[float, float],
        crop_lon_range: tuple[float, float]
):
    # Open the dataset
    with xr.open_dataset(datasetLocation, engine='netCDF4') as ds:
        # Subset the dataset
        ds = ds.sel(
            lat=slice(*lat_range),
            lon=slice(*lon_range),
            drop=True
        )

        # Crop the dataset if necessary
        if crop_lat_range or crop_lon_range:
            ds = crop(ds, lat_range=crop_lat_range, lon_range=crop_lon_range)

    return ds


def combineECCO2Datasets(
        datasetLocations: list[str],
        lat_range: tuple[float, float] = (5, 23.75),
        lon_range: tuple[float, float] = (245, 267.5),
        crop_lat_range: tuple[float, float] = (18, 25),
        crop_lon_range: tuple[float, float] = (240, 270)
):
    # Combine the datasets
    for datasetLocation in datasetLocations:
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
