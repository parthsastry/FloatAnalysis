"""
This script computes the net gradient of the mean background salinity
at density surfaces over the grid.

It uses the density and salinity computed from the
climatology dataset, uses a running mean over a 1 year window to
compute a mean field and computes the zonal and meridional gradients
at the density surfaces.

Use after densityMap.py on the climatology dataset.

AUTHOR: Parth Sastry
"""

import xarray as xr
import numpy as np
from geopy import distance
import argparse


def flagLocations(
        climatology: xr.Dataset,
) -> xr.Dataset:
    """
    Assign flag to locations in climatology dataset depending on
    which difference method to use for computing gradients.

    FLAGS:
        0 - Forward Difference
        1 - Central Difference
        2 - Backward Difference
    Parameters:
        climatology (xr.Dataset): The climatology dataset with density and
                                  salinity.
    Returns:
        xr.Dataset: Dataset with flagged locations.
    """

    # Initialize flags
    flaggedClimatology = climatology.copy()

    flaggedClimatology['zonalFlag'] = xr.DataArray(
        np.full_like(flaggedClimatology['SA'].data[0, :, :, 0], np.nan),
        dims=flaggedClimatology['SA'].dims[1:3],
        coords={
            'LONGITUDE': flaggedClimatology['LONGITUDE'],
            'LATITUDE': flaggedClimatology['LATITUDE']
        }
    )
    flaggedClimatology['meridionalFlag'] = xr.DataArray(
        np.full_like(flaggedClimatology['SA'].data[0, :, :, 0], np.nan),
        dims=flaggedClimatology['SA'].dims[1:3],
        coords={
            'LONGITUDE': flaggedClimatology['LONGITUDE'],
            'LATITUDE': flaggedClimatology['LATITUDE']
        }
    )

    for lon in flaggedClimatology['LONGITUDE'].data:
        for lat in flaggedClimatology['LATITUDE'].data:

            if (flaggedClimatology['SA'].sel(
                    LONGITUDE=lon, LATITUDE=lat
                 ).isnull().any()):
                # Skip if the data is missing for this location
                continue

            if (lon == flaggedClimatology['LONGITUDE'].data[0]):
                # Forward Difference for zonal gradient
                flaggedClimatology['zonalFlag'].loc[
                    dict(LONGITUDE=lon, LATITUDE=lat)
                ] = 0
            elif (lon == flaggedClimatology['LONGITUDE'].data[-1]):
                # Backward Difference for zonal gradient
                flaggedClimatology['zonalFlag'].loc[
                    dict(LONGITUDE=lon, LATITUDE=lat)
                ] = 2
            else:
                if (flaggedClimatology['SA'].sel(
                        LONGITUDE=lon-1, LATITUDE=lat
                     ).isnull().all()):
                    # Forward Difference for zonal gradient
                    flaggedClimatology['zonalFlag'].loc[
                        dict(LONGITUDE=lon, LATITUDE=lat)
                    ] = 0
                elif (flaggedClimatology['SA'].sel(
                        LONGITUDE=lon+1, LATITUDE=lat
                      ).isnull().all()):
                    # Backward Difference for zonal gradient
                    flaggedClimatology['zonalFlag'].loc[
                        dict(LONGITUDE=lon, LATITUDE=lat)
                    ] = 2
                else:
                    # Central Difference for zonal gradient
                    flaggedClimatology['zonalFlag'].loc[
                        dict(LONGITUDE=lon, LATITUDE=lat)
                    ] = 1

            if (lat == flaggedClimatology['LATITUDE'].data[0]):
                # Forward Difference for meridional gradient
                flaggedClimatology['meridionalFlag'].loc[
                    dict(LONGITUDE=lon, LATITUDE=lat)
                ] = 0

            elif (lat == flaggedClimatology['LATITUDE'].data[-1]):
                # Backward Difference for meridional gradient
                flaggedClimatology['meridionalFlag'].loc[
                    dict(LONGITUDE=lon, LATITUDE=lat)
                ] = 2
            else:
                if (flaggedClimatology['SA'].sel(
                        LONGITUDE=lon, LATITUDE=lat-1
                     ).isnull().all()):
                    # Forward Difference for meridional gradient
                    flaggedClimatology['meridionalFlag'].loc[
                        dict(LONGITUDE=lon, LATITUDE=lat)
                    ] = 0
                elif (flaggedClimatology['SA'].sel(
                        LONGITUDE=lon, LATITUDE=lat+1
                      ).isnull().all()):
                    # Backward Difference for meridional gradient
                    flaggedClimatology['meridionalFlag'].loc[
                        dict(LONGITUDE=lon, LATITUDE=lat)
                    ] = 2
                else:
                    # Central Difference for meridional gradient
                    flaggedClimatology['meridionalFlag'].loc[
                        dict(LONGITUDE=lon, LATITUDE=lat)
                    ] = 1

    # Return the dataset with flagged locations
    return flaggedClimatology


def computeGradient(
        flaggedClimatology: xr.Dataset,
) -> xr.Dataset:
    """
    Compute the net gradient of the mean background salinity at density
    surfaces.

    Parameters:
        flaggedClimatology (xr.Dataset): The climatology dataset with density
                                  and salinity and the locations flagged
                                  for zonal and meridional gradients.

    Returns:
        xr.Dataset: Dataset with computed gradients.
    """

    # Create a new dataset with a rolling mean over a 1 year window
    # This will be used to compute the mean background salinity gradient

    climGradient = flaggedClimatology[
        ['SA', 'sigma0', 'zonalFlag', 'meridionalFlag']
    ].rolling(
        TIME=12, center=True, min_periods=5
    ).mean(dim='TIME')
    climGradient['gradient'] = xr.DataArray(
        np.full_like(flaggedClimatology['SA'].data, np.nan),
        dims=flaggedClimatology['SA'].dims,
        coords=flaggedClimatology['SA'].coords
    )

    for lon in climGradient['LONGITUDE'].data:
        for lat in climGradient['LATITUDE'].data:

            print(f"Computing gradient at lon: {lon}, lat: {lat}")

            # Check if valid lat/lon
            if climGradient['sigma0'].sel(
                    LONGITUDE=lon, LATITUDE=lat
            ).isnull().all():
                continue

            zonalFlag = climGradient['zonalFlag'].sel(
                LONGITUDE=lon, LATITUDE=lat
            ).data
            meridionalFlag = climGradient['meridionalFlag'].sel(
                LONGITUDE=lon, LATITUDE=lat
            ).data

            for time in climGradient['TIME'].data:

                # Calculate the zonal and meridional gradients at the
                # density surface
                dens = climGradient['sigma0'].sel(
                    LONGITUDE=lon, LATITUDE=lat, TIME=time
                )
                salt = climGradient['SA'].sel(
                    LONGITUDE=lon, LATITUDE=lat, TIME=time
                )

                # First - zonal Gradient
                if (zonalFlag == 1):
                    # Central Difference for zonal gradient
                    zonalGradient = 0.5*(
                        ((salt - np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon-1, LATITUDE=lat, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon-1, LATITUDE=lat, TIME=time
                            ).values
                        )) /
                            distance.distance((lat, lon), (lat, lon-1)).km)
                        + ((np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon+1, LATITUDE=lat, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon+1, LATITUDE=lat, TIME=time
                            ).values
                        ) - salt) /
                            distance.distance((lat, lon+1), (lat, lon)).km)
                    )

                elif (zonalFlag == 0):
                    # Forward Difference for zonal gradient
                    zonalGradient = ((
                        np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon+1, LATITUDE=lat, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon+1, LATITUDE=lat, TIME=time
                            ).values
                        ) - salt) /
                        distance.distance((lat, lon+1), (lat, lon)).km)

                elif (zonalFlag == 2):
                    # Backward Difference for zonal gradient
                    zonalGradient = ((
                        salt - np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon-1, LATITUDE=lat, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon-1, LATITUDE=lat, TIME=time
                            ).values
                        )) /
                        distance.distance((lat, lon), (lat, lon-1)).km)

                # Now - Meridional Gradient
                if (meridionalFlag == 1):
                    # Central Difference for meridional gradient
                    meridionalGradient = 0.5*(
                        ((salt - np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon, LATITUDE=lat-1, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon, LATITUDE=lat-1, TIME=time
                            ).values
                        )) /
                            distance.distance((lat, lon), (lat-1, lon)).km)
                        + ((np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon, LATITUDE=lat+1, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon, LATITUDE=lat+1, TIME=time
                            ).values
                        ) - salt) /
                            distance.distance((lat+1, lon), (lat, lon)).km)
                    )

                elif (meridionalFlag == 0):
                    # Forward Difference for meridional gradient
                    meridionalGradient = ((
                        np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon, LATITUDE=lat+1, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon, LATITUDE=lat+1, TIME=time
                            ).values
                        ) - salt) /
                        distance.distance((lat+1, lon), (lat, lon)).km)

                elif (meridionalFlag == 2):
                    # Backward Difference for meridional gradient
                    meridionalGradient = ((
                        salt - np.interp(
                            dens.values,
                            climGradient['sigma0'].sel(
                                LONGITUDE=lon, LATITUDE=lat-1, TIME=time
                            ).values,
                            climGradient['SA'].sel(
                                LONGITUDE=lon, LATITUDE=lat-1, TIME=time
                            ).values
                        )) /
                        distance.distance((lat, lon), (lat-1, lon)).km)

                # Calculate the net gradient
                netGradient = np.sqrt(
                    meridionalGradient**2 + zonalGradient**2
                )

                climGradient['gradient'].loc[
                    dict(
                        LONGITUDE=lon,
                        LATITUDE=lat,
                        TIME=time
                    )] = netGradient

    # Return the dataset with computed gradients
    return climGradient


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute gradients of mean background"
                    " salinity at density surfaces."
    )

    parser.add_argument(
        '--climatology_file',
        type=str,
        help='Path to the climatology dataset file.'
    )
    args = parser.parse_args()

    # Load the climatology dataset
    print(f"Loading climatology dataset from {args.climatology_file}")
    try:
        climatology = xr.open_dataset(args.climatology_file)
    except FileNotFoundError:
        print(f"Error: File {args.climatology_file} not found.")
        exit(1)
    print(f"Loaded climatology dataset from {args.climatology_file}")

    # Flag locations in the climatology dataset
    print("Flagging locations in climatology dataset...")
    flaggedClimatology = flagLocations(climatology)
    print("Flagged locations in climatology dataset.")

    # Compute the gradients
    print("Computing gradients...")
    gradients = computeGradient(flaggedClimatology)
    print("Computed gradients.")

    # Save the gradients to a new file
    gradients.to_netcdf(
        args.climatology_file.replace(
            '_densityMapped_minimal.nc', '_computed_gradients.nc'),
        mode='w', format='NETCDF4'
    )
