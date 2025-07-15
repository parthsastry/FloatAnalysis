"""
This script computes the salinity anomaly for each profile

This involves a rolling-mean of the monthly climatology and
interpolating the climatology to the profile locations to compute
the background field.

We then subtract the background field from the profile salinity
to get the anomaly.
"""

import xarray as xr
import numpy as np


def computeAnomalies(
        climatology: xr.Dataset,
        profile: xr.Dataset,
        MLDclimatology: xr.Dataset
) -> xr.Dataset:
    """
    Interpolate the climatology to the profile location and time.
    Compute the salinity anomaly by subtracting the climatology
    from the profile salinity.
    Also add in the climatological MLD to the dataset for writing.

    Parameters:
        climatology (xr.Dataset): The climatology dataset.
        profile (xr.Dataset): The profile dataset.
        MLDclimatology (xr.Dataset): The climatological MLD dataset.

    Returns:
        xr.Dataset: Profile dataset with anomalies
    """

    # Interpolate climatology to profile locations
    profileLocationClimatology = climatology[['SA', 'sigma0 ']].interp(
        LATITUDE=('casts', profile['lat'].data),
        LONGITUDE=('casts', (profile['lon'].data + 360) % 360),
        kwargs={'fill_value': None}
    )
    profileLocationClimatology['casts'] = ("casts", profile['casts'].data)

    # Get cast numbers where interpolation is missing
    missingInterpolationCasts = profileLocationClimatology['casts'].data[
        np.isnan(profileLocationClimatology['SA'].data)[0, :, 0]
    ]

    # Iterate over all casts, look at density surfaces for the particular cast,
    # interpolate the climatology to the density surfaces for the month of the
    # cast, and a year around the cast month to get the background field.
    # Then subtract the background field from the profile salinity to get the
    # anomaly.

    # Create a new dataset to hold the anomalies and the interpolated and
    # smoothed climatology, store the MLD from climatological MLD dataset
    salinityAnomaly = profile.copy()
    salinityAnomaly['SA_anomaly'] = xr.DataArray(
        np.full_like(profile['SA'].data, np.nan),
        dims=profile['SA'].dims,
        coords=profile['SA'].coords
    )
    salinityAnomaly['SA_climatology'] = xr.DataArray(
        np.full_like(profile['SA'].data, np.nan),
        dims=profile['SA'].dims,
        coords=profile['SA'].coords
    )
    salinityAnomaly.drop_vars(['SA'])

    # QUESTION - can this be done without a loop? Think later

    for cast in profileLocationClimatology['casts'][:]:

        castNum = cast.values

        if np.isin(castNum, missingInterpolationCasts):
            # If the cast is missing interpolation, skip it
            print(f"Missing interpolation for cast {castNum}, skipping.")
            continue

        # Get density surfaces and salinity for the current cast
        castSigma0 = profile['sigma0'].sel(casts=castNum)
        castSA = profile['SA'].sel(casts=castNum)

        castYear = profile['time.year'].sel(casts=castNum).data
        castMonth = profile['time.month'].sel(casts=castNum).data

        # Get the climatology for the current cast
        castClimatology = profileLocationClimatology.sel(
            casts=castNum,
            TIME=slice(
                np.datetime64(f'{castYear}-{castMonth:02d}') -
                np.timedelta64(6, 'M'),
                np.datetime64(f'{castYear}-{castMonth:02d}') +
                np.timedelta64(6, 'M')
            )
        )

        # Create a numpy array to hold the mean interpolated background
        meanInterpolatedBackground = \
            np.full_like(castSA.data, np.nan)

        for time in castClimatology['TIME'].data:
            # Get the background field for the current time
            backgroundField = castClimatology.sel(TIME=time)

            try:
                stackedArrs = np.stack(
                    (meanInterpolatedBackground, np.interp(
                        castSigma0.data,
                        backgroundField['sigma0'].data,
                        backgroundField['SA'].data,
                        left=np.nan, right=np.nan
                    )), axis=1
                )
            except ValueError:
                print(castNum)
                print(castSigma0.data.shape)
                print(backgroundField['sigma0'].data.shape)
                print(backgroundField['SA'].data.shape)
                raise
            # Compute the mean of the interpolated background field
            np.nanmean(stackedArrs, axis=1, out=meanInterpolatedBackground)

        if np.isnan(meanInterpolatedBackground).all():
            # If all values are NaN, skip this cast
            continue

        # Store the interpolated and smoothed climatology
        salinityAnomaly['SA_climatology'].loc[dict(casts=castNum)] = (
            meanInterpolatedBackground
        )

        # Compute and store the anomaly
        salinityAnomaly['SA_anomaly'].loc[dict(casts=castNum)] = (
            castSA - meanInterpolatedBackground
        )

    # Return the dataset with anomalies
    return salinityAnomaly


if __name__ == "__main__":
    # Load climatology and profile datasets
    climatology = xr.open_dataset(
        "../data/ARGO_RG2019_Climatology/subsetClimatology/"
        "RG_ArgoClim_ETP_densityMapped_minimal.nc"
    )
    profile = xr.open_dataset(
        "../data/ARGO_VortexProfiles/subsetProfiles/"
        "EasternTropicalPacific_densityMapped_minimal.nc"
    )

    # Compute anomalies
    anomalies = computeAnomalies(climatology, profile)

    # Save the anomalies to a new NetCDF file
    anomalies.to_netcdf(
        "../data/ARGO_VortexProfiles/subsetProfiles/"
        "EasternTropicalPacific_anomalies.nc"
    )
