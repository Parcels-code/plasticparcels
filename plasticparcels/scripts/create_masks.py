# https://github.com/Parcels-code/GlobalMassBudget/blob/main/create_masks_NEMO0083.py


"""
TODO:
Come back to this on the advice from Daan:

https://www.mercator-ocean.eu/static-files-description/
Mask files can be found in /storage2/oceanparcels/data/input_data/MOi/domain_ORCA0083-N006/
These are downloaded directly from MOi - where though?


"""
import numpy as np

import os
import glob
import xarray as xr

from .utils import create_directory, load_settings


def NEMO_select_section(extent, lon, lat, val):

    lon_mean = .5*(extent[0]+extent[1])
    lat_mean = .5*(extent[2]+extent[3])

    i_min = np.unravel_index((np.sqrt((lon-lon_mean)**2 + (lat-lat_mean)**2)).argmin(), lon.shape)

    i_lon_s = np.where((lon[i_min[0], :] > extent[0]) & (lon[i_min[0], :] < extent[1]))[0][0]
    i_lon_e = np.where((lon[i_min[0], :] > extent[0]) & (lon[i_min[0], :] < extent[1]))[0][-1]

    i_lat_s = np.where((lat[:, i_min[1]] > extent[2]) & (lat[:, i_min[1]] < extent[3]))[0][0]
    i_lat_e = np.where((lat[:, i_min[1]] > extent[2]) & (lat[:, i_min[1]] < extent[3]))[0][-1]

    return lon[i_lat_s:i_lat_e, i_lon_s:i_lon_e], lat[i_lat_s:i_lat_e, i_lon_s:i_lon_e], val[i_lat_s:i_lat_e, i_lon_s:i_lon_e]


def to_netcdf(output_filename, data, data_name, lons, lats, explanation=''):
    """
    All data is written to netcdf files to speed up computations
    """
    dict_data = {}
    for data_, name_ in zip(data, data_name):
        dict_data[name_] = (["x", "y"], data_)
    dict_data['explanation'] = explanation

    dict_coords = {}
    dict_coords['lon'] = (["x", "y"], lons)
    dict_coords['lat'] = (["x", "y"], lats)

    ds = xr.Dataset(
        dict_data,
        coords=dict_coords,
    )
    ds.to_netcdf(output_filename)


def get_mask_land(field, lons, lats, outfile='./tmp_mask_land'):
    """
    Mask with true on land cells, false on ocean cells
    """
    if os.path.exists(outfile):
        ds = xr.open_dataset(outfile)
        mask_land = np.array(ds['mask_land'], dtype=bool)
    else:
        mask_land = np.isnan(field)
        to_netcdf(outfile, [mask_land], ['mask_land'], lons, lats, explanation='land mask')
    return mask_land


def get_shore_nodes(landmask):
    """Function that detects the shore nodes, i.e. the land nodes directly
    next to the ocean. Computes the Laplacian of landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype('int')

    return shore


def get_shore_nodes_diagonal(landmask):
    """Function that detects the shore nodes, i.e. the land nodes where
    one of the 8 nearest nodes is ocean. Computes the Laplacian of landmask
    and the Laplacian of the 45 degree rotated landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap += np.roll(landmask, (-1, 1), axis=(0, 1)) + np.roll(landmask, (1, 1), axis=(0, 1))
    mask_lap += np.roll(landmask, (-1, -1), axis=(0, 1)) + np.roll(landmask, (1, -1), axis=(0, 1))
    mask_lap -= 8*landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype('int')

    return shore


def create_displacement_field(landmask, lons, lats, double_cell=False, outfile='./tmp_mask_coast'):
    """Function that creates a displacement field 1 m/s away from the shore.

    - landmask: the land mask dUilt using `make_landmask`.
    - double_cell: Boolean for determining if you want a double cell.
      Default set to False.

    Output: two 2D arrays, one for each camponent of the velocity.
    """
    if os.path.exists(outfile):
        ds = xr.open_dataset(outfile)
        v_x = np.array(ds['land_current_u'], dtype=float)
        v_y = np.array(ds['land_current_v'], dtype=float)

    else:
        shore = get_shore_nodes(landmask)
        shore_d = get_shore_nodes_diagonal(landmask)  # bordering ocean directly and diagonally
        shore_c = shore_d - shore  # corner nodes that only border ocean diagonally

        Ly = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)  # Simple derivative
        Lx = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)

        Ly_c = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)
        Ly_c += np.roll(landmask, (-1, -1), axis=(0, 1)) + np.roll(landmask, (-1, 1), axis=(0, 1))  # Include y-component of diagonal neighbours
        Ly_c += - np.roll(landmask, (1, -1), axis=(0, 1)) - np.roll(landmask, (1, 1), axis=(0, 1))

        Lx_c = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)
        Lx_c += np.roll(landmask, (-1, -1), axis=(1, 0)) + np.roll(landmask, (-1, 1), axis=(1, 0))  # Include x-component of diagonal neighbours
        Lx_c += - np.roll(landmask, (1, -1), axis=(1, 0)) - np.roll(landmask, (1, 1), axis=(1, 0))

        v_x = -Lx*(shore)
        v_y = -Ly*(shore)

        v_x_c = -Lx_c*(shore_c)
        v_y_c = -Ly_c*(shore_c)

        v_x = v_x + v_x_c
        v_y = v_y + v_y_c

        magnitude = np.sqrt(v_y**2 + v_x**2)
        # the coastal nodes between land create a problem. Magnitude there is zero
        # I force it to be 1 to avoid problems when normalizing.
        ny, nx = np.where(magnitude == 0)
        magnitude[ny, nx] = 1

        v_x = v_x/magnitude
        v_y = v_y/magnitude

        to_netcdf(outfile, [v_x, v_y], ['land_current_u', 'land_current_v'], lons, lats, explanation='land current, pusing particles on land back to the sea, magnitude of 1')
    return v_x, v_y


def get_mask_coast(mask_land, lons, lats, outfile='./tmp_mask_coast'):
    """
    calculate the coast mask. With coastal cells, we mean cells in the water, adjacent to land
    """
    if os.path.exists(outfile):
        ds = xr.open_dataset(outfile)
        mask_coast = np.array(ds['mask_coast'], dtype=bool)
    else:
        # check the upper,lower,left & right neighbor: if one of these is an ocean cell, set to landborder
        mask_coast = ~mask_land & (np.roll(mask_land, 1, axis=0) | np.roll(mask_land, -1, axis=0)
                                   | np.roll(mask_land, 1, axis=1) | np.roll(mask_land, -1, axis=1))
        to_netcdf(outfile, [mask_coast], ['mask_coast'], lons, lats, explanation='coast mask (ocean cells adjacent to land')
    return mask_coast


# Load settings
settings_filename = 'default_settings.json'
settings = load_settings(settings_filename)
modelname = settings['ocean']['modelname']

# Create a directory to save the mask files
save_directory = '../data/masks/'
create_directory(save_directory)

# Use temperature field to create a land-mask
dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])
dirread_mesh = os.path.join(settings['ocean']['directory'], settings['ocean']['ocean_mesh'])
sst_filename = glob.glob(dirread_model+'T*')[0]  # Use the first temperature file that pops up

data_T = xr.open_dataset(sst_filename)
data_grid = xr.open_dataset(dirread_mesh, decode_times=False)

# NaN SST on the top layer are land points, and non-NaN SST are ocean points
sst = data_T['votemper'].values[0, :, :]  # Take top level of temperature dataset
lons = data_grid['glamt'][0, :, :].values  # explicitly use the t-cell points
lats = data_grid['gphit'][0, :, :].values

# Create land mask
mask_land = get_mask_land(sst, lons, lats, outfile=save_directory+'mask_land_'+modelname+'.nc')

# Create a 1m/s velocity normal to the coastlines for a possible unbeaching kernel
v_x, v_y = create_displacement_field(mask_land*1, lons, lats, outfile=save_directory+'land_current_'+modelname+'.nc')

# Create a coast mask
mask_coast = get_mask_coast(mask_land, lons, lats, outfile=save_directory+'mask_coast_'+modelname+'.nc')
