"""
Script to map pressure/latitude/longitude datasets containing temperature
and salinity data to density levels using the Gibbs SeaWater (GSW)
Oceanographic Toolbox of TEOS-10

This script is designed to work with datasets that I have pre-processed,
but the variables/dimensions can be adjusted to work with other datasets.

AUTHOR: Parth Sastry
"""

import xarray as xr
import gsw_xarray as gsw_xr
import argparse


def addPotDensity(
        dataset: xr.Dataset,
        type: str = 'Profile'
) -> xr.Dataset:
    """
    Add Potential Density to the dataset.

    Parameters:
        dataset (xr.Dataset): Input dataset containing temperature data.
        type (str): Type of dataset ('Profile' or 'Climatology').

    Returns:
        xr.Dataset: Dataset with Potential Density added.
    """

    if type.lower() == 'profile':
        # For profile data, use following variable names
        lat = dataset['lat']
        lon = dataset['lon']
        temp = dataset['temperature']
        salt = dataset['salinity']
        pres = gsw_xr.p_from_z(
            z=-dataset['z'],
            lat=lat
        )

    elif type.lower() == 'climatology':
        # For climatology data, add anomaly to mean values
        lat = dataset['LATITUDE']
        lon = dataset['LONGITUDE']
        temp = dataset['ARGO_TEMPERATURE_MEAN'] + \
            dataset['ARGO_TEMPERATURE_ANOMALY']
        salt = dataset['ARGO_SALINITY_MEAN'] + \
            dataset['ARGO_SALINITY_ANOMALY']
        pres = dataset['PRESSURE']

    # July 9, 2025 Discussion: Refer TEOS-10 documentation and slides
    # for justification of using sigma_0 instead of other density
    # conversions.
    # We're using Conservative Temperature (CT) and Absolute Salinity (SA)
    # for density calculations

    # An alternate choice could be to use sigma_1, referenced to 1000 dbar,
    # but we should be fine with sigma_0, which is referenced to 0 dbar.

    # Calculate Absolute Salinity
    SA = gsw_xr.SA_from_SP(
        SP=salt,
        p=pres,
        lat=lat,
        lon=lon
    )
    dataset['SA'] = SA

    # Calculate Conservative Temperature
    CT = gsw_xr.CT_from_t(SA=SA, t=temp, p=pres)
    dataset['CT'] = CT

    # Calculate Potential Density Anomaly
    dataset['sigma0'] = gsw_xr.sigma0(SA=SA, CT=CT)

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Add Potential Density to a dataset using GSW"
                    " Oceanographic Toolbox."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input NetCDF file containing temperature'
             ' and salinity data.'
    )
    parser.add_argument(
        'type',
        type=str,
        help='Type of input dataset (Profile/Climatology).'
             ' Used to determine variable names and methods'
    )

    args = parser.parse_args()

    # Load the dataset
    ds = xr.open_dataset(args.input_file)

    # Add Potential Density to the dataset
    ds_with_density = addPotDensity(
        dataset=ds,
        type=args.type
    )

    # Save the modified dataset to a new NetCDF file
    output_file = args.input_file.replace('.nc', '_densityMapped.nc')
    ds_with_density.to_netcdf(
        output_file, mode='w', format='NETCDF4'
    )

    # Minimal dataset to store salinity and density
    minimal_ds = ds_with_density[['SA', 'sigma0']]
    minimal_ds.to_netcdf(
        output_file.replace('.nc', '_minimal.nc'), mode='w', format='NETCDF4'
    )
