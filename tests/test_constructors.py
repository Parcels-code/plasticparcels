import plasticparcels as pp
import parcels
from datetime import datetime, timedelta
import pytest


def make_simple_fieldset():
    from parcels._datasets.structured.generated import simple_UV_dataset
    ds = simple_UV_dataset()

    # TODO can be removed after Parcels v4.0.3 is released
    ds.lon.attrs["units"] = "degrees_east"
    ds.lat.attrs["units"] = "degrees_north"

    return parcels.FieldSet.from_sgrid_conventions(ds)


def make_standard_simulation_settings():
    simulation_settings = {'startdate': datetime.strptime('2020-01-04-00:00:00', '%Y-%m-%d-%H:%M:%S'),
                           'runtime': timedelta(days=2),
                           'outputdt': timedelta(hours=1),
                           'dt': timedelta(minutes=20),
                           }
    return simulation_settings


def make_standard_plastictype_settings():
    plastictype_settings = {'wind_coefficient': 0.01,    # Percentage of wind to apply to particles
                            'plastic_diameter': 0.01,    # Plastic particle diameter (m)
                            'plastic_density': 1030.,    # Plastic particle density (kg/m^3)
                            }
    return plastictype_settings


# Test the create_hydrodynamic_fieldset() function
def test_create_hydrodynamic_fieldset():
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)
    settings['simulation'] = make_standard_simulation_settings()

    fieldset = pp.constructors.create_hydrodynamic_fieldset(settings)

    assert isinstance(fieldset, parcels.FieldSet)


# Test the create_fieldset() function
@pytest.mark.parametrize('use_3D', [True, False])
@pytest.mark.parametrize('use_biofouling', [True, False])
@pytest.mark.parametrize('use_stokes', [True, False])
@pytest.mark.parametrize('use_wind', [True, False])
@pytest.mark.parametrize('use_mixing', [True, False])
def test_create_fieldset(use_3D, use_biofouling, use_stokes, use_wind, use_mixing):
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()

    settings['use_3D'] = use_3D
    settings['use_biofouling'] = use_biofouling
    settings['use_stokes'] = use_stokes
    settings['use_wind'] = use_wind
    settings['use_mixing'] = use_mixing

    fieldset = pp.constructors.create_fieldset(settings)

    assert isinstance(fieldset, parcels.FieldSet)


# Test the base create_particleset() function
@pytest.mark.parametrize('plastic_amount', [None, 1])
def test_create_particleset(plastic_amount):
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()

    settings['plastictype'] = make_standard_plastictype_settings()

    if plastic_amount is None:
        release_locations = {'lons': [18], 'lats': [35]}
    else:
        release_locations = {'lons': [18], 'lats': [35], 'plastic_amount': [plastic_amount]}

    fieldset = make_simple_fieldset()
    pset = pp.constructors.create_particleset(fieldset, settings, release_locations)

    assert isinstance(pset, parcels.ParticleSet)


# Test three different initialisation maps
@pytest.mark.parametrize('initialisation_map', ['coastal', 'fisheries', 'rivers'])
def test_create_particleset_from_map(initialisation_map):
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()

    settings['plastictype'] = make_standard_plastictype_settings()

    settings['release'] = {'initialisation_type': initialisation_map,
                           'country': 'Italy',
                           }

    fieldset = make_simple_fieldset()
    pset = pp.constructors.create_particleset_from_map(fieldset, settings)

    assert isinstance(pset, parcels.ParticleSet)


# Test the two global concentration release map types
@pytest.mark.parametrize('concentration_type', ['Beach', 'Ocean'])
def test_create_particleset_from_map_concentrations(concentration_type):
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()

    settings['plastictype'] = make_standard_plastictype_settings()

    settings['release'] = {'initialisation_type': 'global_concentrations',
                           'concentration_type': concentration_type,
                           }

    fieldset = make_simple_fieldset()
    pset = pp.constructors.create_particleset_from_map(fieldset, settings)

    assert isinstance(pset, parcels.ParticleSet)


# Test three different initialisation maps
@pytest.mark.parametrize('use_3D', [True, False])
@pytest.mark.parametrize('use_biofouling', [True, False])
@pytest.mark.parametrize('use_stokes', [True, False])
@pytest.mark.parametrize('use_wind', [True, False])
@pytest.mark.parametrize('use_mixing', [True, False])
def test_create_kernel(use_3D, use_biofouling, use_stokes, use_wind, use_mixing):
    settings_file = 'tests/test_data/test_settings.json'
    settings = pp.utils.load_settings(settings_file)
    settings = pp.utils.download_plasticparcels_dataset('NEMO0083', settings, 'input_data')

    settings['simulation'] = make_standard_simulation_settings()

    settings['use_3D'] = use_3D
    settings['use_biofouling'] = use_biofouling
    settings['use_stokes'] = use_stokes
    settings['use_wind'] = use_wind
    settings['use_mixing'] = use_mixing

    fieldset = pp.constructors.create_fieldset(settings)
    kernels = pp.constructors.create_kernel(fieldset)

    if use_3D:
        assert parcels.kernels.AdvectionRK4_3D in kernels
        assert pp.kernels.checkThroughBathymetry in kernels
        assert pp.kernels.checkErrorThroughSurface in kernels
    else:
        assert parcels.kernels.AdvectionRK4 in kernels

    if use_3D and use_biofouling:
        assert pp.kernels.Biofouling in kernels
    elif use_3D and not use_biofouling:
        assert pp.kernels.SettlingVelocity in kernels

    if use_mixing:
        assert pp.kernels.VerticalMixing in kernels

    if use_stokes:
        assert pp.kernels.StokesDrift in kernels
        assert pp.kernels.unbeaching in kernels

    if use_wind:
        assert pp.kernels.WindageDrift in kernels
        assert pp.kernels.unbeaching in kernels

    assert pp.kernels.periodicBC in kernels
    assert pp.kernels.PolyTEOS10_bsq in kernels
