import numpy as np
from glob import glob
import pandas as pd
from datetime import timedelta
import os
import shapely as sh
import json
from pathlib import Path
from urllib.request import urlretrieve
import warnings


def getclosest_ij(lats, lons, latpt, lonpt):
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats - latpt) ** 2 + (lons - lonpt) ** 2  # find squared distance of every point on grid
    minindex_flattened = np.nanargmin(dist_sq)  # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened,
                            lats.shape)  # Get 2D index for latvals and lonvals arrays from 1D index


def select_files(dirread, string_, date_current, dt_sim, i_date_s=-13, i_date_e=-3, dt_margin=8):
    """Function to select files in a directory based on a current date and a simulation duration."""
    # set dt_margin to e.g. 32 when dealing with monthly data, or 8 when dealing with weekly data
    yr0 = date_current.year

    time_start = date_current - timedelta(days=dt_margin)
    time_end = date_current + timedelta(days=dt_sim) + timedelta(days=dt_margin)
    yrEnd = time_end.year

    ragged_files = [glob(dirread + string_ % (yr0+i)) for i in range(-1, yrEnd-yr0+1)]
    possible_files = []

    for i in range(len(ragged_files)):
        for file_ in ragged_files[i]:
            possible_files.append(file_)

    possible_files = sorted(possible_files)

    indices_use = []
    for i1, file_ in enumerate(possible_files):
        str_date = os.path.basename(file_)[i_date_s:i_date_e]
        date_ = pd.Timestamp(str_date)

        if date_ > time_start and date_ < time_end:
            indices_use.append(i1)

    files_use = list(possible_files[i] for i in indices_use)
    return files_use


def create_directory(directory):
    """Function to create a directory if it does not exist."""
    if not os.path.exists(directory):
        print('Creating directory %s' % directory)
        os.makedirs(directory)


def distance(lon1, lat1, lon2, lat2):
    """Function to calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to Radians:
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2r, lon1r)
    dlat = np.subtract(lat2r, lat1r)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1r),
                           np.multiply(np.cos(lat2r),
                                       np.power(np.sin(np.divide(dlon, 2)), 2))))

    c = np.multiply(2, np.arcsin(np.sqrt(a)))
    r = 6371

    return c*r


def get_coords_from_polygon(shape):
    """Function to return a list of coordinate points on a Polygon
    (or MultiPolygon) shape.

    Based on: https://stackoverflow.com/questions/58844463/how-to-get-a-list-of-every-point-inside-a-multipolygon-using-shapely
    """
    coords = []

    if isinstance(shape, sh.geometry.Polygon):
        coords.append(np.asarray(shape.exterior.coords[:-1]))
        for linearring in shape.interiors:
            coords.append(np.asarray(linearring.coords[:-1]))
    elif isinstance(shape, sh.geometry.MultiPolygon):
        for polygon in shape.geoms:
            coords.append(get_coords_from_polygon(polygon))
    coords = np.concatenate(coords)
    return coords


def load_settings(filename):
    """Function to load a settings file in json format."""
    with open(filename, "r") as file:
        settings = json.load(file)
    return settings


def download_plasticparcels_dataset(dataset: str, settings, data_home=None):
    """Function to load an example dataset from the parcels website.

    This function provides quick access to a small number of example datasets
    that are useful in documentation and testing in parcels.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    settings : dict
        Dictionary containing the settings for the models and simulation.
    data_home : pathlike, optional
        The directory in which to cache data. If not specified, the value
        of the ``PLASTICPARCELS_DATA`` environment variable, if any, is used.
        Otherwise the default location is assigned by :func:`get_data_home`.

    Returns
    -------
    dataset_folder : Path
        Path to the folder containing the downloaded dataset files.
    """
    plasticparcels_data_files = {
        "NEMO0083": [
            (("release_maps", "coastal"), "coastal_population_MPW_NEMO0083.csv"),
            (("release_maps", "rivers"), "river_emissions_NEMO0083.csv"),
            (("release_maps", "fisheries"), "agg_data_fisheries_info_NEMO0083.csv"),
            (("release_maps", "global_concentrations"), "global_concentrations_NEMO0083.csv"),
            (("unbeaching", "filename"), "land_current_NEMO0083.nc"),
        ],
    }

    plasticparcels_data_url = "https://plasticadrift.science.uu.nl/plasticparcels/"

    if dataset not in plasticparcels_data_files:
        raise ValueError(
            f"Dataset {dataset!r} not found. Available datasets are: "
            ", ".join(plasticparcels_data_files.keys())
        )

    if data_home is None:
        data_home = os.getcwd()
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    dataset_folder = os.path.join(data_home, dataset)

    if not Path(dataset_folder).exists():
        Path(dataset_folder).mkdir(parents=True, exist_ok=True)

    for settings_path, filename in plasticparcels_data_files[dataset]:
        filepath = os.path.join(dataset_folder, filename)
        try:
            settings[settings_path[0]][settings_path[1]] = filepath
            if not os.path.exists(filepath):
                url = f"{plasticparcels_data_url}/{dataset}/{filename}"
                urlretrieve(url, str(filepath))
        except Exception as e:
            warnings.warn(f"Could not set settings path {settings_path} for dataset {dataset!r}.", stacklevel=1)
    return settings
