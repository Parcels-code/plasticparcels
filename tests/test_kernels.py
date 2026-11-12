import plasticparcels as pp
import parcels
import numpy as np
from datetime import datetime, timedelta
import pytest


def make_standard_simulation_settings():
    simulation_settings = {'startdate': datetime.strptime('2020-01-04-00:00:00', '%Y-%m-%d-%H:%M:%S'),
                           'runtime': timedelta(days=1),
                           'outputdt': timedelta(hours=1),
                           'dt': timedelta(minutes=20),
                           }
    return simulation_settings


def make_standard_plastictype_settings():
    # Use tiny wind percentage because test data set is not large and wind speeds are quick!
    plastictype_settings = {'wind_coefficient': 0.0001,     # Percentage of wind to apply to particles
                            'plastic_diameter': 0.001,  # Plastic particle diameter (m)
                            'plastic_density': 1027.67,   # Plastic particle density (kg/m^3)
                            }
    return plastictype_settings


def make_standard_particleset(fieldset, settings):
    # Generate a particleset that has particles in the test domain
    release_locations = {'lons': [18, 18.25, 18.5], 'lats': [35, 35, 35],
                         'plastic_amount': [1, 1, 1]}
    pset = pp.constructors.create_particleset(fieldset, settings, release_locations)

    return pset

def checkBelowDataDepth(particles, fieldset):
    # The vertical mixing kernel can push particles below the test dataset depth, throwing an
    # out of bounds error. This kernel will keep particles above the max depth.
    ptcls = particles[particles.z + particles.dz >= fieldset.max_depth]
    if len(ptcls) > 0:
        # move a meter above the max depth
        ptcls.dz = fieldset.max_depth - ptcls.z - 1.0 # noqa
        ptcls.state = parcels.StatusCode.Success


@pytest.mark.parametrize('use_3D', [True, False])
def test_advection_only(use_3D):
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    # Turn on/off 3D advection
    settings['use_3D'] = use_3D

    # Turn off all other processes
    settings['use_biofouling'] = False
    settings['use_stokes'] = False
    settings['use_wind'] = False
    settings['use_mixing'] = False

    fieldset = pp.constructors.create_fieldset(settings)
    if use_3D:
        kernels = [parcels.kernels.AdvectionRK4_3D, pp.kernels.checkThroughBathymetry,
                   pp.kernels.checkErrorThroughSurface,]
    else:
        kernels = [parcels.kernels.AdvectionRK4]

    pset = make_standard_particleset(fieldset, settings)

    start_lons = pset.x.copy()
    start_lats = pset.y.copy()

    pset.execute(kernels, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])

    # Assert that the particles move from their initial location
    assert (np.sum(np.abs(pset.x - start_lons)) > 0.) & (np.sum(np.abs(pset.y - start_lats)) > 0.)


def test_settling_velocity():
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    # Turn on 3D advection
    settings['use_3D'] = True

    # Turn off all other processes
    settings['use_biofouling'] = False
    settings['use_stokes'] = False
    settings['use_wind'] = False
    settings['use_mixing'] = False

    fieldset = pp.constructors.create_fieldset(settings)

    kernels = pp.constructors.create_kernel(fieldset)

    pset = make_standard_particleset(fieldset, settings)

    start_lons = pset.x.copy()
    start_lats = pset.y.copy()
    start_depths = pset.z.copy()

    pset.execute(kernels, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])

    # Assert that the particles move from their initial location
    assert (np.sum(np.abs(pset.x - start_lons)) > 0.) & (np.sum(np.abs(pset.y - start_lats)) > 0.) & (np.sum(np.abs(pset.z - start_depths)) > 0.)


def test_biofouling():
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    # Turn on biofouling
    settings['use_biofouling'] = True

    # Turn off all other processes
    settings['use_3D'] = False
    settings['use_stokes'] = False
    settings['use_wind'] = False
    settings['use_mixing'] = False

    fieldset = pp.constructors.create_fieldset(settings)

    kernels = [pp.kernels.PolyTEOS10_bsq, pp.kernels.Biofouling,
               pp.kernels.checkThroughBathymetry, pp.kernels.checkErrorThroughSurface]

    pset = make_standard_particleset(fieldset, settings)

    start_depths = pset.z.copy()

    print(settings['simulation'])
    pset.execute(kernels, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])

    # Assert that the particles move from their initial location
    assert (np.sum(np.abs(pset.z - start_depths)) > 0.)


def test_Stokes():
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    # Required for the unbeaching kernel
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    # Turn on Stokes Drift
    settings['use_stokes'] = True

    # Turn off all other processes
    settings['use_3D'] = False
    settings['use_biofouling'] = False
    settings['use_wind'] = False
    settings['use_mixing'] = False

    fieldset = pp.constructors.create_fieldset(settings)

    kernels = [pp.kernels.StokesDrift, pp.kernels.unbeaching, pp.kernels.periodicBC]

    pset = make_standard_particleset(fieldset, settings)

    start_lons = pset.x.copy()
    start_lats = pset.y.copy()
    pset.execute(kernels, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])

    # Assert that the particles move from their initial location
    assert (np.sum(np.abs(pset.x - start_lons)) > 0.) & (np.sum(np.abs(pset.y - start_lats)) > 0.)


def test_wind():
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    # Required for the unbeaching kernel
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    # Turn on Stokes Drift
    settings['use_wind'] = True

    # Turn off all other processes
    settings['use_3D'] = False
    settings['use_biofouling'] = False
    settings['use_stokes'] = False
    settings['use_mixing'] = False

    fieldset = pp.constructors.create_fieldset(settings)

    kernels = [pp.kernels.WindageDrift, pp.kernels.unbeaching, pp.kernels.periodicBC]

    pset = make_standard_particleset(fieldset, settings)

    start_lons = pset.x.copy()
    start_lats = pset.y.copy()

    pset.execute(kernels, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])

    # Assert that the particles move from their initial location
    assert (np.sum(np.abs(pset.x - start_lons)) > 0.) & (np.sum(np.abs(pset.y - start_lats)) > 0.)


def test_mixing():
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    settings['use_3D'] = True
    settings['use_mixing'] = True

    # Turn off all other processes
    settings['use_wind'] = False
    settings['use_biofouling'] = False
    settings['use_stokes'] = False

    fieldset = pp.constructors.create_fieldset(settings)
    fieldset.add_context('max_depth', fieldset.U.grid.depth[-1])

    # Set the simulation runtime to just 1 day so particles aren't kicked around significantly
    settings['simulation']['runtime'] = timedelta(days=1)

    kernels = [parcels.kernels.AdvectionRK4_3D, pp.kernels.checkThroughBathymetry,
               pp.kernels.checkErrorThroughSurface]

    kernels_mixing = [parcels.kernels.AdvectionRK4_3D, pp.kernels.VerticalMixing,
                      checkBelowDataDepth, pp.kernels.checkThroughBathymetry,
                      pp.kernels.checkErrorThroughSurface]

    pset = make_standard_particleset(fieldset, settings)
    pset_mixing = make_standard_particleset(fieldset, settings)
    pset.execute(kernels, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])
    pset_mixing.execute(kernels_mixing, runtime=settings['simulation']['runtime'], dt=settings['simulation']['dt'])

    # Assert that the particles move from their initial location
    assert (np.sum(np.abs(pset.x - pset_mixing.x)) > 0.) & (np.sum(np.abs(pset.y - pset_mixing.y)) > 0.)

def test_TEOSdensity_kernels():
    """ Adapted test from Parcels v3 codebase.
    """
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)

    settings['simulation'] = make_standard_simulation_settings()
    settings['plastictype'] = make_standard_plastictype_settings()

    # Turn off all other processes
    settings['use_3D'] = False
    settings['use_biofouling'] = False
    settings['use_stokes'] = False
    settings['use_wind'] = False
    settings['use_mixing'] = False

    fieldset = pp.constructors.create_fieldset(settings)
    fieldset.add_constant_field("absolute_salinity", 30.)
    fieldset.add_constant_field("conservative_temperature", 10.)

    release_locations = {'lons': [5], 'lats': [5], 'depths': [1000],
                         'plastic_amount': [1]}

    pset = pp.constructors.create_particleset(fieldset, settings, release_locations)

    pset.execute(pp.kernels.PolyTEOS10_bsq, runtime=1, dt=1)

    assert np.allclose(pset[0].seawater_density, 1027.45140)
