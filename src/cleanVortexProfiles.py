"""
This script cleans and subsets datasets from Simoes-Sousa et al. (2025)'s
background and matched profiles data.

It processes temperature and salinity profiles, removes unnecessary variables,
and subsets the data based on specified latitude and longitude ranges.

Defaults to my primary region of interest, the Eastern Tropical Pacific.

AUTHOR: Parth Sastry
"""

import xarray as xr
import numpy as np
import argparse

from helperLibrary import subset, crop


def cleanProfiles(
        dataset: xr.Dataset,
        type: str = 'background',
        variable: str = 'temperature'
) -> xr.Dataset:
    """
    Clean the profile dataset by removing unnecessary variables
    and adding in dummy variables (if needed).

    Parameters:
        dataset (xr.Dataset): The input dataset.
        type (str): The type of profile dataset, 'matched' or 'background'.
        variable (str): Variable in dataset, 'temperature' or 'salinity'.

    Returns:
        xr.Dataset: The cleaned profile dataset.
    """
    # Add dummy variable to background dataset
    if type == 'background':
        dataset['eddy_rotation'] = xr.DataArray(
            np.zeros_like(dataset['lat']),
            dims=dataset['lat'].dims,
            coords=dataset['lat'].coords
        )

    # Remove unnecessary variables
    dataset = dataset[[variable, 'eddy_rotation']]

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Clean and subset Vortex Profiles data.'
    )
    parser.add_argument(
        '--region', type=str,
        default='EasternTropicalPacific',
        help='Region to subset the data for.'
    )
    parser.add_argument(
        '--lat_range', type=float, nargs=2,
        default=(5, 23.75),
        help='Latitude range for subsetting (min, max).'
    )
    parser.add_argument(
        '--lon_range', type=float, nargs=2,
        default=(-115, -92.5),
        help='Longitude range for subsetting (min, max).'
    )
    parser.add_argument(
        '--crop_lat_range', type=float, nargs=2,
        default=None,
        help='Latitude range for cropping (min, max).'
    )
    parser.add_argument(
        '--crop_lon_range', type=float, nargs=2,
        default=None,
        help='Longitude range for cropping (min, max).'
    )

    args = parser.parse_args()

    path = '../data/ARGO_VortexProfiles/'
    idx = ['background', 'matched']
    vars = ['temperature', 'salinity']

    profiles = {var: {} for var in vars}

    for var in vars:
        for id in idx:

            ds = xr.open_dataset(f'{path}/{id}/{id}_pfl_{var.capitalize()}.nc')
            ds = cleanProfiles(ds, type=id, variable=var)
            profiles[var][id] = ds

        # Subset data to region
        profiles[var] = subset(
            profiles[var],
            var_names=['lat', 'lon', 'time'],
            lat_range=args.lat_range,
            lon_range=args.lon_range,
            # Set time range to after 2004 to use RG Climatology
            time_range=(
                np.datetime64('2004-01-01'), np.datetime64('2023-12-31')
            )
        )

        # Crop data (if needed)
        if args.crop_lat_range is not None and args.crop_lon_range is not None:
            profiles[var] = crop(
                profiles[var],
                var_names=['lat', 'lon', 'time'],
                lat_range=args.crop_lat_range,
                lon_range=args.crop_lon_range
            )

        # Merge background and matched datasets
        profiles[var] = profiles[var]['background'].merge(
            profiles[var]['matched']
        ).sortby('time')

    # Merge temperature and salinity datasets
    profiles = xr.merge(
        [profiles['temperature'], profiles['salinity']],
        join='inner'
    ).sortby('time')

    # Save (if not already saved)
    profiles.to_netcdf(
        f'{path}/subsetProfiles/' +
        f'{args.region}.nc', mode='w', format='NETCDF4'
    )
