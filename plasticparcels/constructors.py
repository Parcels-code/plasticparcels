import os
import numpy as np

import pandas as pd
from parcels import FieldSet, Field, ParticleSet, JITParticle, Variable, AdvectionRK4, AdvectionRK4_3D
from parcels.tools.converters import Geographic, GeographicPolar

from plasticparcels.kernels import PolyTEOS10_bsq, StokesDrift, WindageDrift, SettlingVelocity, Biofouling, VerticalMixing, unbeaching, periodicBC, checkErrorThroughSurface, deleteParticle, checkThroughBathymetry
from plasticparcels.utils import select_files


def create_hydrodynamic_fieldset(settings):
    """A constructor method to create a `Parcels.Fieldset` from hydrodynamic
    model data.

    Parameters
    ----------
    settings :
        A dictionary of settings that contains an ocean model directory, a filename style,
        and the location of the ocean model mesh file, used to create the fieldset.

    Returns
    -------
    fieldset
        A `parcels.FieldSet` object.
    """
    # Location of hydrodynamic data
    dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])

    # Start date and runtime of the simulation
    startdate = settings['simulation']['startdate']
    runtime = int(np.ceil(settings['simulation']['runtime'].total_seconds()/86400.))  # convert to days

    # Mesh masks
    ocean_mesh = os.path.join(settings['ocean']['directory'], settings['ocean']['ocean_mesh'])  # mesh_mask

    # Setup input for fieldset creation
    ufiles = select_files(dirread_model, 'U_%4i*.nc', startdate, runtime, dt_margin=3)
    vfiles = select_files(dirread_model, 'V_%4i*.nc', startdate, runtime, dt_margin=3)
    wfiles = select_files(dirread_model, 'W_%4i*.nc', startdate, runtime, dt_margin=3)
    tfiles = select_files(dirread_model, 'T_%4i*.nc', startdate, runtime, dt_margin=3)
    sfiles = select_files(dirread_model, 'S_%4i*.nc', startdate, runtime, dt_margin=3)

    filenames = {'U': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': wfiles},
                 'conservative_temperature': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': tfiles},
                 'absolute_salinity': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': sfiles}}

    variables = settings['ocean']['variables']
    dimensions = settings['ocean']['dimensions']
    indices = settings['ocean']['indices']

    if not settings['use_3D']:
        indices['depth'] = range(0, 2)

    # Load the fieldset
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                                  indices=indices, allow_time_extrapolation=settings['allow_time_extrapolation'])

    # Create flags for custom particle behaviour
    fieldset.add_constant('use_mixing', settings['use_mixing'])
    fieldset.add_constant('use_biofouling', settings['use_biofouling'])
    fieldset.add_constant('use_stokes', settings['use_stokes'])
    fieldset.add_constant('use_wind', settings['use_wind'])
    fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-1]
    fieldset.add_constant('use_3D', settings['use_3D'])

    # Add in bathymetry
    fieldset.add_constant('z_start', 0.5)
    bathymetry_variables = settings['ocean']['bathymetry_variables']
    bathymetry_dimensions = settings['ocean']['bathymetry_dimensions']
    bathymetry_mesh = os.path.join(settings['ocean']['directory'], settings['ocean']['bathymetry_mesh'])
    bathymetry_field = Field.from_netcdf(bathymetry_mesh, bathymetry_variables, bathymetry_dimensions)
    fieldset.add_field(bathymetry_field)

    # If vertical mixing is turned on, add in the KPP-Profile
    if fieldset.use_mixing:
        dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])
        kzfiles = select_files(dirread_model, 'KZ_%4i*.nc', startdate, runtime, dt_margin=3)
        mixing_filenames = {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': kzfiles}
        mixing_variables = settings['ocean']['vertical_mixing_variables']
        mixing_dimensions = settings['ocean']['vertical_mixing_dimensions']
        mixing_fieldset = FieldSet.from_nemo(mixing_filenames, mixing_variables, mixing_dimensions)
        fieldset.add_field(mixing_fieldset.mixing_kz)  # phytoplankton primary productivity

    return fieldset


def create_fieldset(settings):
    """A constructor method to create a `Parcels.Fieldset` with all fields
    necessary for a plasticparcels simulation (e.g., a hydrodynamic model
    velocity field, biogeochemical model variable fields, a wind field, etc.).

    Parameters
    ----------
    settings :
        A dictionary of settings that contains an ocean model information,
        biogeochemical model information, wind model information,
        and other optional settings.

    Returns
    -------
    fieldset
        A `parcels.FieldSet` object.
    """
    # First create the hydrodynamic fieldset
    fieldset = create_hydrodynamic_fieldset(settings)

    # Now add the other fields
    # Start date and runtime of the simulation
    startdate = settings['simulation']['startdate']
    runtime = int(np.ceil(settings['simulation']['runtime'].total_seconds()/86400.))  # convert to days

    if fieldset.use_biofouling:
        # MOi glossary: https://www.mercator-ocean.eu/wp-content/uploads/2021/11/Glossary.pdf
        # and https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-028.pdf

        # Add BGC constants to current fieldset
        for key in settings['bgc']['constants']:
            fieldset.add_constant(key, settings['bgc']['constants'][key])

        # Create a fieldset with BGC data
        dirread_bgc = os.path.join(settings['bgc']['directory'], settings['bgc']['filename_style'])
        bgc_mesh = os.path.join(settings['bgc']['directory'], settings['bgc']['bgc_mesh'])  # mesh_mask_4th

        dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])
        wfiles = select_files(dirread_model, 'W_%4i*.nc', startdate, runtime, dt_margin=3)

        ppfiles = select_files(dirread_bgc, 'nppv_%4i*.nc', startdate, runtime, dt_margin=8)
        phy1files = select_files(dirread_bgc, 'phy_%4i*.nc', startdate, runtime, dt_margin=8)
        phy2files = select_files(dirread_bgc, 'phy2_%4i*.nc', startdate, runtime, dt_margin=8)

        filenames_bio = {'pp_phyto': {'lon': bgc_mesh, 'lat': bgc_mesh, 'depth': wfiles[0], 'data': ppfiles},
                         'bio_nanophy': {'lon': bgc_mesh, 'lat': bgc_mesh, 'depth': wfiles[0], 'data': phy1files},
                         'bio_diatom': {'lon': bgc_mesh, 'lat': bgc_mesh, 'depth': wfiles[0], 'data': phy2files}}

        variables_bio = settings['bgc']['variables']
        dimensions_bio = settings['bgc']['dimensions']

        # Create the BGC fieldset
        bio_fieldset = FieldSet.from_nemo(filenames_bio, variables_bio, dimensions_bio)

        # Add the fields to the main fieldset
        fieldset.add_field(bio_fieldset.pp_phyto)  # phytoplankton primary productivity
        fieldset.add_field(bio_fieldset.bio_nanophy)  # nanopyhtoplankton concentration [mmol C m-3]
        fieldset.add_field(bio_fieldset.bio_diatom)  # diatom concentration [mmol C m-3]

    if fieldset.use_stokes:
        dirread_Stokes = os.path.join(settings['stokes']['directory'], settings['stokes']['filename_style'])
        wavesfiles = select_files(dirread_Stokes, '%4i*.nc', startdate, runtime, dt_margin=32)

        filenames_Stokes = {'Stokes_U': wavesfiles,
                            'Stokes_V': wavesfiles,
                            'wave_Tp': wavesfiles}

        variables_Stokes = settings['stokes']['variables']
        dimensions_Stokes = settings['stokes']['dimensions']

        fieldset_Stokes = FieldSet.from_netcdf(filenames_Stokes, variables_Stokes, dimensions_Stokes, mesh='spherical')
        fieldset_Stokes.Stokes_U.units = GeographicPolar()
        fieldset_Stokes.Stokes_V.units = Geographic()
        fieldset_Stokes.add_periodic_halo(zonal=True)

        fieldset.add_field(fieldset_Stokes.Stokes_U)
        fieldset.add_field(fieldset_Stokes.Stokes_V)
        fieldset.add_field(fieldset_Stokes.wave_Tp)

    if fieldset.use_wind:
        dirread_wind = os.path.join(settings['wind']['directory'], settings['wind']['filename_style'])
        windfiles = select_files(dirread_wind, '%4i*.nc', startdate, runtime, dt_margin=32)

        filenames_wind = {'Wind_U': windfiles,
                          'Wind_V': windfiles}

        variables_wind = settings['wind']['variables']
        dimensions_wind = settings['wind']['dimensions']

        fieldset_wind = FieldSet.from_netcdf(filenames_wind, variables_wind, dimensions_wind, mesh='spherical')
        fieldset_wind.Wind_U.units = GeographicPolar()
        fieldset_wind.Wind_V.units = Geographic()
        fieldset_wind.add_periodic_halo(zonal=True)

        fieldset.add_field(fieldset_wind.Wind_U)
        fieldset.add_field(fieldset_wind.Wind_V)

    # Apply unbeaching currents when Stokes/Wind can push particles into land cells
    if fieldset.use_stokes or fieldset.use_wind > 0:
        unbeachfiles = settings['unbeaching']['filename']
        filenames_unbeach = {'unbeach_U': unbeachfiles,
                             'unbeach_V': unbeachfiles}

        variables_unbeach = settings['unbeaching']['variables']

        dimensions_unbeach = settings['unbeaching']['dimensions']

        fieldset_unbeach = FieldSet.from_netcdf(filenames_unbeach, variables_unbeach, dimensions_unbeach, mesh='spherical')
        fieldset_unbeach.unbeach_U.units = GeographicPolar()
        fieldset_unbeach.unbeach_V.units = Geographic()

        fieldset.add_field(fieldset_unbeach.unbeach_U)
        fieldset.add_field(fieldset_unbeach.unbeach_V)

    fieldset.add_constant('verbose_delete', settings['verbose_delete'])

    return fieldset


def create_particleset(fieldset, settings, release_locations):
    """A constructor method to create a `Parcels.ParticleSet` for a
    `plasticparcels` simulation.

    Parameters
    ----------
    fieldset :
        A `Parcels.FieldSet` object.
    settings :
        A dictionary containing the plastic-type settings.
    release_locations :
        A dictionary containing release locations for particles.

    Returns
    -------
    particleset
        A parcels.ParticleSet object.
    """
    # Set the longitude, latitude, and plastic amount per particle
    lons = np.array(release_locations['lons'])
    lats = np.array(release_locations['lats'])
    if 'plastic_amount' in release_locations.keys():
        plastic_amounts = release_locations['plastic_amount']
    else:
        plastic_amounts = np.full_like(lons, np.nan)

    if 'depths' in release_locations.keys():
        depths = release_locations['depths']
    else:
        depths = np.full_like(lons, 0.)

    # Set particle properties
    plastic_densities = np.full(lons.shape, settings['plastictype']['plastic_density'])
    plastic_diameters = np.full(lons.shape, settings['plastictype']['plastic_diameter'])
    wind_coefficients = np.full(lons.shape, settings['plastictype']['wind_coefficient'])

    PlasticParticle = JITParticle
    variables = [Variable('plastic_diameter', dtype=np.float32, initial=np.nan, to_write=False),
                 Variable('plastic_density', dtype=np.float32, initial=np.nan, to_write=False),
                 Variable('wind_coefficient', dtype=np.float32, initial=0., to_write=False),
                 Variable('settling_velocity', dtype=np.float64, initial=0., to_write=False),
                 Variable('seawater_density', dtype=np.float32, initial=np.nan, to_write=False),
                 Variable('absolute_salinity', dtype=np.float64, initial=np.nan, to_write=False),
                 Variable('algae_amount', dtype=np.float64, initial=0., to_write=False),
                 Variable('plastic_amount', dtype=np.float32, initial=0., to_write=True)]

    for variable in variables:
        setattr(PlasticParticle, variable.name, variable)

    pset = ParticleSet.from_list(fieldset,
                                 PlasticParticle,
                                 lon=lons,
                                 lat=lats,
                                 depth=depths,
                                 plastic_diameter=plastic_diameters,
                                 plastic_density=plastic_densities,
                                 wind_coefficient=wind_coefficients,
                                 plastic_amount=plastic_amounts)

    return pset


def create_particleset_from_map(fieldset, settings):
    """A constructor method to create a `Parcels.ParticleSet` for a
    `plasticparcels` simulation using one of the available initialisation maps.

    Parameters
    ----------
    fieldset :
        A `Parcels.FieldSet` object.
    settings :
        A dictionary containing release settings and plastic-type settings.

    Returns
    -------
    particleset
        A `parcels.ParticleSet` object.
    """
    # Load release type information
    release_type = settings['release']['initialisation_type']

    release_quantity_names = {
        'coastal': 'MPW_Cell',
        'rivers': 'Emissions',
        'fisheries': 'fishing_hours',
        'global_concentrations': 'Concentration'
    }
    release_quantity_name = release_quantity_names[release_type]

    particle_locations = pd.read_csv(settings['release_maps'][release_type])

    # Select specific continent/region/subregion/country/economic status if applicable:
    if 'continent' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Continent'] == settings['release']['continent']]
    if 'region' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Region'] == settings['release']['region']]
    if 'subregion' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Subregion'] == settings['release']['subregion']]
    if 'country' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Country'] == settings['release']['country']]
    if 'economicstatus' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Economic status'] == settings['release']['economicstatus']]
    if 'concentration_type' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['ConcentrationType'] == settings['release']['concentration_type']]

    particle_locations = particle_locations.groupby(['Longitude', 'Latitude'])[release_quantity_name].agg('sum').reset_index()
    particle_locations = particle_locations[particle_locations[release_quantity_name] > 0]

    release_locations = {'lons': particle_locations['Longitude'],
                         'lats': particle_locations['Latitude'],
                         'plastic_amount': particle_locations[release_quantity_name]}

    pset = create_particleset(fieldset, settings, release_locations)

    return pset


def create_kernel(fieldset):
    """A constructor method to create a list of kernels for a `plasticparcels`
    simulation.

    Parameters
    ----------
    fieldset :
        A `parcels.FieldSet` object containing constants used to turn on/off
        different kernel behaviours.

    Returns
    -------
    kernels :
        A list of kernels used in the execution of the particle set.
    """
    kernels = []
    kernels.append(PolyTEOS10_bsq)  # To set the seawater_density variable

    if fieldset.use_3D:
        kernels.append(AdvectionRK4_3D)
    else:
        kernels.append(AdvectionRK4)

    if not fieldset.use_biofouling and fieldset.use_3D:
        kernels.append(SettlingVelocity)
    elif fieldset.use_biofouling and fieldset.use_3D:  # Must be in 3D to use biofouling mode
        kernels.append(Biofouling)
    elif fieldset.use_biofouling and not fieldset.use_3D:
        print('Biofouling mode is only available in 3D mode. Please set use_3D\
              to True in the settings file.')

    if fieldset.use_stokes:
        kernels.append(StokesDrift)
    if fieldset.use_wind:
        kernels.append(WindageDrift)

    if fieldset.use_mixing:
        kernels.append(VerticalMixing)

    # Add the unbeaching kernel
    if fieldset.use_stokes or fieldset.use_wind:
        kernels.append(unbeaching)

    if fieldset.use_3D: # Add statuscode kernels for 3D advection
        kernels.append(checkThroughBathymetry)
        kernels.append(checkErrorThroughSurface)

    # Add statuscode kernels
    kernels.append(periodicBC)
    kernels.append(deleteParticle)

    return kernels
