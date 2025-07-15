import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature


def subset(
    datasets: dict[str, xr.Dataset] | xr.Dataset,
    var_names: list[str] = None,
    lat_range: tuple[float, float] = None,
    lon_range: tuple[float, float] = None,
    time_range: tuple[np.datetime64, np.datetime64] = None
) -> dict[str, xr.Dataset]:
    """
    Subset xarray Dataset(s) based on latitude and longitude ranges.

    Parameters:
        datasets (dict[str, xr.Dataset] | xr.Dataset): Dataset(s) to subset.
        lat_range (tuple[float, float], optional): Latitude range (min, max).
        lon_range (tuple[float, float], optional): Longitude range (min, max).

    Returns:
        dict[str, xr.Dataset]: Dictionary of subsetted datasets.
        Note: dict key will be 'dataset' if a single xr.Dataset is provided.
    """
    subsetted_datasets = {}
    if isinstance(datasets, xr.Dataset):
        singleItem = True
        datasets = {'dataset': datasets}

    # Create dummy ranges if not provided
    ranges = {
        var_names[0]: lat_range if lat_range is not None else (-90, 90),
        var_names[1]: lon_range if lon_range is not None else (-180, 180),
        var_names[2]: time_range if time_range is not None else
        (np.datetime64('1970-01-01'), np.datetime64('2026-01-01'))
    }

    if lat_range is None and lon_range is None and time_range is None:
        # If no ranges are provided, return the original datasets
        return datasets

    for key, ds in datasets.items():
        subsetted_datasets[key] = ds.where(
            (ds[var_names[0]] >= ranges[var_names[0]][0]) &
            (ds[var_names[0]] <= ranges[var_names[0]][1]) &
            (ds[var_names[1]] >= ranges[var_names[1]][0]) &
            (ds[var_names[1]] <= ranges[var_names[1]][1]) &
            (ds[var_names[2]] >= ranges[var_names[2]][0]) &
            (ds[var_names[2]] <= ranges[var_names[2]][1]),
            drop=True
        )

    if singleItem:
        # If a single dataset was provided, return it directly
        return subsetted_datasets['dataset']

    return subsetted_datasets


def crop(
        datasets: dict[str, xr.Dataset] | xr.Dataset,
        var_names: list[str] = None,
        lat_range: tuple[float, float] = None,
        lon_range: tuple[float, float] = None,
        time_range: tuple[np.datetime64, np.datetime64] = None
) -> dict[str, xr.Dataset]:
    """
    Crop xarray Dataset(s) based on
    latitude, longitude and time ranges.

    Parameters:
        datasets (dict[str, xr.Dataset] | xr.Dataset): Dataset(s) to crop
        lat_range (tuple[float, float]): Latitude range (min, max)
        lon_range (tuple[float, float]): Longitude range (min, max)
        time_range (tuple[np.datetime64, np.datetime64]): Time range (min, max)

    Returns:
        dict[str, xr.Dataset]: Dictionary of dataset(s) with subset removed
        Note: dict key will be 'dataset' if a single xr.Dataset is provided
    """
    cropped_datasets = {}
    if isinstance(datasets, xr.Dataset):
        singleItem = True
        datasets = {'dataset': datasets}

    # Create dummy ranges if not provided
    ranges = {
        var_names[0]: lat_range if lat_range is not None else (0, 0),
        var_names[1]: lon_range if lon_range is not None else (0, 0),
        var_names[2]: time_range if time_range is not None else
        (np.datetime64('2026-01-01'), np.datetime64('2026-01-01'))
    }

    if lat_range is None and lon_range is None and time_range is None:
        # If no ranges are provided, return the original datasets
        return datasets

    if time_range is not None:
        for key, ds in datasets.items():
            cropped_datasets[key] = ds.where(
                (ds[var_names[0]] <= ranges[var_names[0]][0]) |
                (ds[var_names[0]] >= ranges[var_names[0]][1]) |
                (ds[var_names[1]] <= ranges[var_names[1]][0]) |
                (ds[var_names[1]] >= ranges[var_names[1]][1]) |
                (ds[var_names[2]] <= ranges[var_names[2]][0]) |
                (ds[var_names[2]] >= ranges[var_names[2]][1]),
                drop=True
            )
    else:
        for key, ds in datasets.items():
            cropped_datasets[key] = ds.where(
                (ds[var_names[0]] <= ranges[var_names[0]][0]) |
                (ds[var_names[0]] >= ranges[var_names[0]][1]) |
                (ds[var_names[1]] <= ranges[var_names[1]][0]) |
                (ds[var_names[1]] >= ranges[var_names[1]][1]),
                drop=True
            )

    if singleItem:
        # If a single dataset was provided, return it directly
        return cropped_datasets['dataset']

    return cropped_datasets


def profileLocationPlot(
        ax: plt.Axes,
        data: xr.Dataset
) -> None:
    """
    Plot the locations of profiles on a map.

    Parameters:
        ax (plt.Axes): Matplotlib Axes object to plot on.
        data (xr.Dataset): xarray Dataset containing profile locations.

    Returns:
        None: The function modifies the Axes object in place.
    """
    labels = {
        -1.0: 'Anticyclones',
        1.0: 'Cyclones',
        0.0: 'Background'
    }
    scatter = ax.scatter(
        data['lon'], data['lat'],
        c=data['eddy_rotation'], cmap='coolwarm', s=1
    )
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.gridlines(
        draw_labels=True, linewidth=0.5, color='gray',
        alpha=0.5, linestyle='dotted'
    )
    ax.set_title(
        f'Profile Locations - Total: {int(data["eddy_rotation"].count())},' +
        f' Cyclones: {int((data["eddy_rotation"] == 1.0).sum())},' +
        f' Anticyclones: {int((data["eddy_rotation"] == -1.0).sum())}'
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(
        handles=scatter.legend_elements()[0],
        labels=[labels[-1.0], labels[0.0], labels[1.0]]
    )
