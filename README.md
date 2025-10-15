## plasticparcels
`plasticparcels` is a python package for simulating the transport and dispersion of plastics in the ocean.

The tool is based on the [`Parcels`](https://parcels-code.org/) computational Lagrangian ocean analysis framework ([@Lange2017](http://dx.doi.org/10.5194/gmd-10-4175-2017) and [@Delandmeter2019](http://dx.doi.org/10.5194/gmd-12-3571-2019)), providing a modular and customisable collection of methods, notebooks, and tutorials for advecting virtual plastic particles with a wide range of physical properties. A peer reviewed paper describing `plasticparcels` can be found here: [![DOI](https://joss.theoj.org/papers/10.21105/joss.07094/status.svg)](https://doi.org/10.21105/joss.07094).

![plasticparcels](docs/_static/plasticparcelslogo.png)

### Installation

`plasticparcels` can be installed using `conda` from the [`conda-forge` channel](https://anaconda.org/conda-forge/plasticparcels) with the following command:

```bash
conda install conda-forge::plasticparcels
```

### Required data

`plasticparcels` has been developed for use with data from the Copernicus Marine Service, and requires the following data to run:

* Hydrodynamic model data: [MOI GLO12 (psy4v3r1)](https://www.mercator-ocean.eu/en/solutions-expertise/accessing-digital-data/product-details/?offer=4217979b-2662-329a-907c-602fdc69c3a3&system=d35404e4-40d3-59d6-3608-581c9495d86a)
* Biogeochemical model data: [MOI BIO4 (biomer4v2r1)](https://www.mercator-ocean.eu/en/solutions-expertise/accessing-digital-data/product-details/?offer=8d0c01f3-81c7-0a59-0d06-602fdf63c5b6&system=dc40b324-7de7-0732-880b-5d9dcf7d344a)
* Wave data: [ECMWF ERA5 Wave](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) (specifically, the variables `mean_wave_period`, `peak_wave_period`, `u_component_stokes_drift`, and `v_component_stokes_drift`.)
* Wind data: [ECMWF ERA5 Wind](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) (specifically, the variables `10m_u_component_of_wind` and `10m_v_component_of_wind`)

For downloading the wind and wave data, we recommend using the [CDS API](https://cds.climate.copernicus.eu/api-how-to).

To run most of the examples, you will need to update the data directories in the relevant settings `.json` file.

Just like the `parcels` framework, `plasticparcels` can be adapted to use other hydrodynamic, biogeochemical, wave, and atmospheric models. If you require assistance, please contact us through the [`plasticparcels` discussions page](https://github.com/Parcels-code/plasticparcels/discussions).

### Community contributions and support
#### Contributing code
We welcome contributions to `plasticparcels`, especially example workbooks and analyses for our [public examples page](https://plastic.parcels-code.org/en/latest/examples.html). To contribute to the project, please submit a [pull request](https://github.com/Parcels-code/plasticparcels/pulls).

#### Requesting features and reporting issues/bugs
If you want to request a new feature, or if you find an issue or bug in the code, please open an issue in the [`plasticparcels` issue tracker](https://github.com/Parcels-code/plasticparcels/issues).

#### Seeking support?
If you would like support using `plasticparcels`, or are have any questions about your `plasticparcels` simulations, please start a discussion in the [`plasticparcels` discussion page](https://github.com/Parcels-code/plasticparcels/discussions).




### Further information
For more information and documentation, see the [plasticparcels documentation](https://plastic.parcels-code.org/).

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07094/status.svg)](https://doi.org/10.21105/joss.07094)
[![unit-tests](https://github.com/Parcels-code/plasticparcels/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/Parcels-code/plasticparcels/actions/workflows/unit_tests.yml)
[![Anaconda-release](https://anaconda.org/conda-forge/plasticparcels/badges/version.svg)](https://anaconda.org/conda-forge/plasticparcels/)
[![Anaconda-date](https://anaconda.org/conda-forge/plasticparcels/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/plasticparcels/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.11388383.svg)](https://doi.org/10.5281/zenodo.11388383)
