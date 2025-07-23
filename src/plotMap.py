"""
This script takes in a minimal profile dataset containing the salinity
anomalies, along with a climatological dataset containing mean salinity
and gradients, and plots the time-averaged mean salinity, rms salinity
anomaly, gradient of the mean salinity field, and mixing length for a
given density.

Requires the bin centers for latitude and longitude to be provided so that
the data can be binned accordingly.

NOTE - Currently hardcoded to work with the Eastern Tropical Pacific
region, but can be adjusted for other regions by changing the extent of
the axes within the plotMap function.

AUTHOR: Parth Sastry
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cmocean
import numpy as np


def plotMap(
        profileAnomalies: xr.Dataset,
        climatologyGradient: xr.Dataset,
        density: float,
        binCenters: tuple[np.ndarray, np.ndarray],
        fig: plt.Figure,
        axs: tuple[GeoAxes, GeoAxes, GeoAxes, GeoAxes],
        salinity_lims: tuple[float, float] = (34.65, 34.80),
        grad_lims: tuple[float, float] = (0, 1.5E-4),
        anomaly_lims: tuple[float, float] = (0, 0.02),
        lambda_lims: tuple[float, float] = (0, 800)
) -> None:
    """
    Plot the mean salinity, rms salinity anomaly, gradient of the mean salinity
    field, and mixing length for a given density.

    Parameters:
        profileAnomalies (xr.Dataset): Profile anomalies dataset.
        climatologyGradient (xr.Dataset): Climatology gradient dataset.
        density (float): Density value for plotting.
        binCenters (tuple[np.ndarray, np.ndarray]): lat/lon bin centers.
        axs (tuple[GeoAxes, GeoAxes, GeoAxes, GeoAxes]): Axes for plotting.
    """

    binSize = binCenters[0][1] - binCenters[0][0]

    anomalies = xr.DataArray(
        np.zeros(profileAnomalies['casts'].size),
        dims='casts',
        coords={'casts': profileAnomalies['casts']}
    )
    for cast in profileAnomalies['casts']:
        anomalies.loc[dict(casts=cast)] = np.interp(
            density,
            profileAnomalies['sigma0'].sel(casts=cast).values,
            profileAnomalies['SA_anomaly'].sel(casts=cast).values
        )

    meanSalinity = xr.DataArray(
        np.zeros((binCenters[0].size, binCenters[1].size)),
        dims=('lon', 'lat'),
        coords={'lon': binCenters[0], 'lat': binCenters[1]}
    )
    meanSalinityGradient = xr.DataArray(
        np.zeros((binCenters[0].size, binCenters[1].size)),
        dims=('lon', 'lat'),
        coords={'lon': binCenters[0], 'lat': binCenters[1]}
    )

    for lon in binCenters[0]:
        for lat in binCenters[1]:

            binSalField = climatologyGradient['SA'].interp(
                LATITUDE=lat,
                LONGITUDE=360+lon,
            )
            binDensityField = climatologyGradient['sigma0'].interp(
                LATITUDE=lat,
                LONGITUDE=360+lon,
            )
            binGradient = climatologyGradient['gradient'].interp(
                LATITUDE=lat,
                LONGITUDE=360+lon,
            )

            # Calculate mean salinity and salinity gradient for the given
            # density
            for time in climatologyGradient['TIME']:
                meanSalinity.loc[lon, lat] += np.interp(
                    density,
                    binDensityField.sel(TIME=time).values,
                    binSalField.sel(TIME=time).values
                )
                meanSalinityGradient.loc[lon, lat] += np.interp(
                    density,
                    binDensityField.sel(TIME=time).values,
                    binGradient.sel(TIME=time).values
                )
            meanSalinity.loc[lon, lat] /= \
                climatologyGradient['TIME'].size
            meanSalinityGradient.loc[lon, lat] /= \
                climatologyGradient['TIME'].size

    # Calculate rms salinity anomaly
    absAnomalies = np.abs(anomalies)

    # Bin the anomalies to the same lat/lon grid as mean salinity
    binnedAnomalies = xr.DataArray(
        np.full((binCenters[0].size, binCenters[1].size), np.nan),
        dims=('lon', 'lat'),
        coords={'lon': binCenters[0], 'lat': binCenters[1]}
    )

    for lon in binCenters[0]:
        for lat in binCenters[1]:

            binnedCasts = profileAnomalies['casts'].where(
                (profileAnomalies['lon'] >= lon - binSize / 2) &
                (profileAnomalies['lon'] < lon + binSize / 2) &
                (profileAnomalies['lat'] >= lat - binSize / 2) &
                (profileAnomalies['lat'] < lat + binSize / 2)
            ).dropna(dim='casts')

            if binnedCasts['casts'].size >= 25:
                binnedAnomalies.loc[lon, lat] = np.sqrt(
                    np.mean(
                        absAnomalies.sel(casts=binnedCasts['casts'])**2
                    )
                )

    # Calculate mixing length
    mixingLength = binnedAnomalies / meanSalinityGradient

    # Plotting the data

    def fmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'${} * 10^{{{}}}$'.format(a, b)

    for ax in axs:
        ax.set_extent([-115, -91, 5, 24.5], crs=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.LAND, edgecolor='black', facecolor='lightgray')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(
            draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
            linestyle='--', xlocs=np.arange(-115, -91, 1.5),
            ylocs=np.arange(5, 24.5, 1.5)
        )

    axs[0].set_title('Mean Salinity')
    meanSalinity.plot(
        x='lon', y='lat',
        ax=axs[0],
        vmin=salinity_lims[0], vmax=salinity_lims[1],
        cmap=cmocean.cm.haline,
        cbar_kwargs={
            'orientation': 'horizontal', 'pad': 0.1,
            'label': r'$\langle \{ \text{S} \} \rangle (\text{g kg}^{-1})$',
            'aspect': 20, 'fraction': 0.05}
    )

    axs[1].set_title("RMS Salinity Anomaly")
    binnedAnomalies.plot(
        x='lon', y='lat',
        ax=axs[1], cmap=cmocean.cm.matter,
        vmin=anomaly_lims[0], vmax=anomaly_lims[1],
        cbar_kwargs={
            'orientation': 'horizontal', 'pad': 0.1,
            'label': r'$\langle \{ S^\prime S^\prime \} \rangle^{0.5}$'
                     r"$ (\text{g kg}^{-1})$",
            'aspect': 20, 'fraction': 0.05}
    )

    axs[2].set_title("Mean Salinity Gradient")
    meanSalinityGradient.plot(
        x='lon', y='lat',
        ax=axs[2], cmap=cmocean.cm.tempo,
        vmin=grad_lims[0], vmax=grad_lims[1],
        cbar_kwargs={
            'orientation': 'horizontal', 'pad': 0.1,
            'label': r'$\langle \nabla \{ \text{S} \} \rangle $'
                     r"$ (\text{g kg}^{-1} \text{km}^{-1})$",
            'format': mticker.FuncFormatter(fmt),
            'aspect': 20, 'fraction': 0.05}
    )

    axs[3].set_title("Mixing Length")
    mixingLength.plot(
        x='lon', y='lat',
        ax=axs[3], cmap=cmocean.cm.amp,
        vmin=lambda_lims[0], vmax=lambda_lims[1],
        cbar_kwargs={
            'orientation': 'horizontal', 'pad': 0.1,
            'label': r'$\lambda (\text{km})$',
            'aspect': 20, 'fraction': 0.05}
    )

    fig.suptitle(
        f"Statistics on the {density:.1f} kg/m³ Isopycnal",
        fontsize=16
    )
