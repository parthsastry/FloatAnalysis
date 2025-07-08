"""
This script merges the Roemmich-Gilson Argo climatology extensions (2019-)
with the base climatology data (2004-18)
"Roemmich, D. and J. Gilson, 2009: The 2004-2008 mean and annual cycle of
temperature, salinity, and steric height in the global ocean from the
Argo Program. Progress in Oceanography, 82, 81-100."

Requires the files -
RG_ArgoClim_Salinity_2019.nc, RG_ArgoClim_Temperature_2019.nc,
and the extension datasets in a subdirectory named 'extensions'.

It reads in the extension datasets one by one, merges them with the base and
saves the merged dataset to a new NetCDF file.

You can also pass in a lat/lon range and a crop range to limit the data,
according to the functions in helperLibrary.py
NOTE - The latitude and longitude values in the climatology are weird.
Latitude extends from -64.5 to 79.5 and longitude from 20.5 to 379.5

Defaults to my primary region of interest, the Eastern Tropical Pacific.

AUTHOR: Parth Sastry
"""

import xarray as xr
import numpy as np
import argparse
from datetime import datetime
import glob

from helperLibrary import subset, crop


def mergeClimatology(
        tempClimatology: xr.Dataset,
        salinityClimatology: xr.Dataset,
        lat_range: tuple[float, float] = None,
        lon_range: tuple[float, float] = None,
        crop_lat_range: tuple[float, float] = None,
        crop_lon_range: tuple[float, float] = None
):
    """
    Merge the temperature and salinity climatology datasets,
    optionally subsetting and cropping them based on lat/lon ranges.

    Parameters:
        tempClimatology (xr.Dataset): Temperature climatology dataset.
        salinityClimatology (xr.Dataset): Salinity climatology dataset.
        lat_range (tuple[float, float]): Latitude range for subsetting.
        lon_range (tuple[float, float]): Longitude range for subsetting.
        crop_lat_range (tuple[float, float]): Latitude range for cropping.
        crop_lon_range (tuple[float, float]): Longitude range for cropping.

    Returns:
        xr.Dataset: Merged climatology dataset.
    """

    # Ensure the time variable is in datetime64 format
    # HACK - This is a workaround - xarray and numpy don't like month to
    # datetime conversion

    timeOrigin = np.datetime64(
        datetime.strptime(
            tempClimatology['TIME'].time_origin, "%d-%b-%Y %H:%M:%S"))
    tempClimatology['TIME'] = timeOrigin.astype('datetime64[M]') + \
        tempClimatology['TIME'].values.astype('timedelta64[M]')

    timeOrigin = np.datetime64(
        datetime.strptime(
            salinityClimatology['TIME'].time_origin, "%d-%b-%Y %H:%M:%S"))
    salinityClimatology['TIME'] = timeOrigin.astype('datetime64[M]') + \
        salinityClimatology['TIME'].values.astype('timedelta64[M]')

    # Subset datasets if lat/lon ranges are provided
    if lat_range or lon_range:
        tempClimatology = subset(
            datasets=tempClimatology, lat_range=lat_range,
            # Ensure the variable names match the climatology datasets
            lon_range=lon_range, var_names=['LATITUDE', 'LONGITUDE', 'TIME']
        )
        salinityClimatology = subset(
            datasets=salinityClimatology, lat_range=lat_range,
            # Ensure the variable names match the climatology datasets
            lon_range=lon_range, var_names=['LATITUDE', 'LONGITUDE', 'TIME']
        )

    # Crop datasets if crop ranges are provided
    if crop_lat_range or crop_lon_range:
        tempClimatology = crop(
            datasets=tempClimatology,
            lat_range=crop_lat_range, lon_range=crop_lon_range,
            var_names=['LATITUDE', 'LONGITUDE', 'TIME']
        )
        salinityClimatology = crop(
            datasets=salinityClimatology,
            lat_range=crop_lat_range, lon_range=crop_lon_range,
            var_names=['LATITUDE', 'LONGITUDE', 'TIME']
        )

    # Clean mean temp and salinity - no need for TIME variable
    tempClimatology['ARGO_TEMPERATURE_MEAN'] = \
        tempClimatology['ARGO_TEMPERATURE_MEAN'].sel(
        TIME=tempClimatology['TIME'].values[0], drop=True
    )
    salinityClimatology['ARGO_SALINITY_MEAN'] = \
        salinityClimatology['ARGO_SALINITY_MEAN'].sel(
        TIME=salinityClimatology['TIME'].values[0], drop=True
    )

    # Clean bathymetry and mapping masks - no need for TIME variable
    tempClimatology['BATHYMETRY_MASK'] = \
        tempClimatology['BATHYMETRY_MASK'].sel(
        TIME=tempClimatology['TIME'].values[0], drop=True
    )
    salinityClimatology['BATHYMETRY_MASK'] = \
        salinityClimatology['BATHYMETRY_MASK'].sel(
        TIME=salinityClimatology['TIME'].values[0], drop=True
    )
    tempClimatology['MAPPING_MASK'] = \
        tempClimatology['MAPPING_MASK'].sel(
        TIME=tempClimatology['TIME'].values[0], drop=True
    )
    salinityClimatology['MAPPING_MASK'] = \
        salinityClimatology['MAPPING_MASK'].sel(
        TIME=salinityClimatology['TIME'].values[0], drop=True
    )

    # Merge the datasets
    merged_climatology = xr.merge([tempClimatology, salinityClimatology])

    return merged_climatology


def mergeExtension(
        extensionDataset: xr.Dataset,
        baseDataset: xr.Dataset,
        lat_range: tuple[float, float] = None,
        lon_range: tuple[float, float] = None,
        crop_lat_range: tuple[float, float] = None,
        crop_lon_range: tuple[float, float] = None
):
    """
    Merge the extension dataset with the base dataset,
    optionally subsetting it based on lat/lon ranges.

    Parameters:
        extensionDataset (xr.Dataset): Extension dataset to merge.
        baseDataset (xr.Dataset): Base dataset to merge with.
        lat_range (tuple[float, float]): Latitude range for subsetting.
        lon_range (tuple[float, float]): Longitude range for subsetting.

    Returns:
        xr.Dataset: Merged dataset.
    """

    # Ensure the time variable is in datetime64 format
    timeOrigin = np.datetime64(
        datetime.strptime(
            extensionDataset['TIME'].time_origin, "%d-%b-%Y %H:%M:%S"))
    extensionDataset['TIME'] = timeOrigin.astype('datetime64[M]') + \
        extensionDataset['TIME'].values.astype('timedelta64[M]')

    # Subset the extension dataset if lat/lon ranges are provided
    if lat_range or lon_range:
        extensionDataset = subset(
            datasets=extensionDataset,
            lat_range=lat_range, lon_range=lon_range,
            var_names=['LATITUDE', 'LONGITUDE', 'TIME'],
        )

    # Crop the extension dataset if crop ranges are provided
    if crop_lat_range or crop_lon_range:
        extensionDataset = crop(
            datasets=extensionDataset,
            lat_range=crop_lat_range, lon_range=crop_lon_range,
            var_names=['LATITUDE', 'LONGITUDE', 'TIME'],
        )

    # Merge the datasets
    merged_dataset = xr.merge([baseDataset, extensionDataset])

    return merged_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Merge Roemmich-Gilson Argo climatology extensions' +
                    ' with base climatology.'
    )
    parser.add_argument(
        '--region', type=str, default='ETP',
        help='Region of interest for subsetting and cropping. ' +
             'Defaults to ETP (Eastern Tropical Pacific).'
    )
    parser.add_argument(
        '--lat_range', type=float, nargs=2,
        default=(5, 23.75),
        help='Latitude range for subsetting (min, max).'
    )
    parser.add_argument(
        '--lon_range', type=float, nargs=2,
        default=(245, 267.5),
        help='Longitude range for subsetting (min, max).'
    )
    parser.add_argument(
        '--crop_lat_range', type=float, nargs=2,
        default=(18, 25),
        help='Latitude range for cropping (min, max).'
    )
    parser.add_argument(
        '--crop_lon_range', type=float, nargs=2,
        default=(260, 270),
        help='Longitude range for cropping (min, max).'
    )
    args = parser.parse_args()

    # Load the base climatology datasets
    path = "../data/ARGO_RG2019_Climatology/"
    temp_climatology = xr.open_dataset(
        path + "RG_ArgoClim_Temperature_2019.nc",
        decode_times=False
    )
    salinity_climatology = xr.open_dataset(
        path + "RG_ArgoClim_Salinity_2019.nc",
        decode_times=False
    )

    climatology = mergeClimatology(
        temp_climatology,
        salinity_climatology,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        crop_lat_range=args.crop_lat_range,
        crop_lon_range=args.crop_lon_range
    )

    extensionFiles = glob.glob(
        path + "extensions/*.nc"
    )

    for ext_file in extensionFiles:
        with xr.open_dataset(ext_file, decode_times=False) as extensionDataset:
            # Merge each extension dataset with the climatology
            print(f"Merging extension dataset: {ext_file}")
            climatology = mergeExtension(
                extensionDataset,
                climatology,
                lat_range=args.lat_range,
                lon_range=args.lon_range,
                crop_lat_range=args.crop_lat_range,
                crop_lon_range=args.crop_lon_range
            )

    # Save the merged climatology dataset
    climatology.to_netcdf(
        path + f"subsetClimatology/RG_ArgoClim_Merged_{args.region}.nc",
        mode='w', format='NETCDF4'
    )
