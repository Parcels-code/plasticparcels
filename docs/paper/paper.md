---
title:  'plasticparcels: A python package for marine plastic dispersal simulations and parameterisation development using parcels'
tags:
  - Plastic dispersal simulations
  - Lagrangian oceanography
  - Fluid dynamics
authors:
  - name: Michael C. Denes
    orcid: 0000-0003-1868-0043
    affiliation: 1
  - name: Erik van Sebille
    orcid: 0000-0003-2041-0704
    affiliation: 1

affiliations:
  - name: Institute for Marine and Atmospheric Research, Utrecht University, the Netherlands
    index: 1
date: 29 May 2024
bibliography: paper.bib
link-bibliography: false
---

# Summary
`plasticparcels` is a python package for simulating the transport and dispersion of plastics in the ocean. The tool is based on `v3.0.3` of the `parcels` computational Lagrangian ocean analysis framework [@Lange2017; @Delandmeter2019], providing a modular and customisable collection of methods, notebooks, and tutorials for advecting virtual plastic particles with a wide range of physical properties. The tool applies a collection of physical processes to the virtual particles, such as Stokes drift, wind-induced drift, biofouling, and turbulent mixing, via custom particle behaviour programmed in the form of `Kernels`. In addition to the fine-scale physics parameterisations, `plasticparcels` provides global particle initialisation maps that represent best estimates for plastic pollution emissions along coastlines [@Jambeck2015], from river sources [@Meijer2021], and in the open-ocean from fishing-related activities [@Kroodsma2018], as well as a current best estimate of buoyant plastic concentrations globally [@Kaandorp2023]. We envisage `plasticparcels` as a tool for easy-to-run plastic dispersal simulations; as well as for rapid prototyping, development, and testing of new fine-scale physics parameterisations.

The current version supports nano- and microplastic behaviour, with support for macroplastics planned in the near-future. It has been designed for use with hydrodynamic and biogeochemical data from the [Copernicus Marine Service](https://marine.copernicus.eu/), providing new plastic modelling capabilities as part of the [NECCTON (New Copernicus Capability for Trophic Ocean Networks)](https://neccton.eu/) project. `plasticparcels` is easily adapted to run on local machines and high-performance computing (HPC) architecture with various hydrodynamic, biogeochemical, and other model fields as input. A future goal is to embed `plasticparcels` within a cloud platform to allow for even more rapid prototyping, development, and simulations.

The core features of `plasticparcels` are: 1) a user-friendly python notebook layer on top of `parcels` that provides a streamlined workflow for performing plastic dispersal simulations, 2) custom `parcels` kernels designed to simulate the fine-scale physical processes that influence the transport of nano- and microplastic particulates, and 3) global particle initialisation maps which represent the best estimate locations of plastic pollution emissions from coastal sources, river sources, open ocean fishing-related activity emission sources, and a current best estimate of buoyant plastic concentrations. We visualise these initialisation maps in Fig. \ref{fig:initialisation_maps}.

![Particle initialisations maps based on a) coastal mismanaged plastic waste emissions [@Jambeck2015], b) riverine mismanaged plastic waste emissions [@Meijer2021], c) fishing activity related plastic emissions [@Kroodsma2018], and d) a current global surface concentration estimate [@Kaandorp2023].\label{fig:initialisation_maps}](initialisation_maps.png){width=100%}

In addition, due to the flexibility of the package, users may use the functions and modular design of `plasticparcels` to enhance their existing `parcels` simulations and workflow. For example, users can use the initialisation maps, associated `ParticleSet` creation methods, and/or the custom physics kernels with their own `parcels` simulations. Post-processing and analysis of the generated trajectory datasets is purposefully left to the user, however some tutorials are provided in the [`plasticparcels` documentation](https://plastic.parcels-code.org/en/latest/examples.html), along with the tutorials in the [`parcels` documentation](https://docs.parcels-code.org/en/latest/documentation.html). We provide an example use case of `plasticparcels` below.

# Statement of need
Marine plastic debris can be found almost everywhere in the ocean. A recent study estimates that there is approximately 3,200 kilotonnes of (initially) positively buoyant plastics in the global ocean in the year 2020 [@Kaandorp2023], where 59-62\% of these plastics are found at the ocean surface, 36-39\% within the deeper ocean, and 1.5-1.9\% along the coastline. They estimate that 500 kilotonnes of positively buoyant plastic enters the ocean each year, where 39-42\% originate from mismanaged waste along coastlines, 45-48\% originate from fishing-related activities (e.g. fishing lines, nets, traps, and crates), and 12-13\% from mismanaged waste entering the ocean via rivers.

Due to its durable, inert, and cheap-to-manufacture nature, plastic has become one of the most abundant manufactured synthetic materials on Earth. Between 1950 and 2017 an estimated 8,300 million tonnes [@Geyer2017] of virgin plastic was produced, with the rate of production only set to increase. Its durability is of primary concern to the marine environment, where, without intervention, they will likely degrade and fragment into smaller pieces that will disperse across ever larger distances. These plastics interact and interfere with marine wildlife, either entangling, or being inadvertently ingested, with documented cases affecting over 900 marine species so far [@Kuhn2020]. To better understand and predict the effects of plastic pollution on the marine environment, it is of paramount importance to map where and how plastic enters our ocean, and the pathways of transport, dispersal patterns, and ultimate fate of these plastics.

Lagrangian ocean analysis, where virtual particles are tracked in hydrodynamic flow fields, is widely used to uncover and investigate the pathways and timescales of the dispersion of plastic particulates in the ocean [@Lebreton2012; @Hardesty2017; @JalonRojas2019; @Chassignet2021; @Kaandorp2023]. However, two important questions arise when performing such Lagrangian simulations. Firstly, what physical processes drive the transport and dispersal of a plastic particle? The properties of plastic particles (e.g., size, shape, and density) determine what the dominant physical processes are at play, and due to the chaotic nature of the ocean, the dispersal patterns and transport behaviours of plastics will critically depend on their properties. Current state-of-the-art ocean models are either too coarse in resolution to capture these processes, or disregard these processes entirely, and so parameterising these processes is important to model and simulate their effects. Secondly, what are the initial release locations and concentrations of marine plastic pollution? Forecasting near-future spatial maps of plastic concentrations is largely an initial value problem, relying on accurate initial conditions for a realistic simulation output. As yet, there is no single comprehensive dataset of estimates for current marine plastic pollution concentrations, mismanaged plastic waste along coastlines, mismanaged plastic waste in rivers, and plastic entering the ocean from fishing-related activities. Combining best estimates of current marine plastic pollution concentrations with likely future plastic pollution sources is necessary for predicting near-future plastic concentration maps in the ocean.

The past decade has seen a growing number of community-developed software packages for performing Lagrangian simulations [@Paris2013; @Fredj2016; @Lange2017; @Doos2017; @Dagestad2018; @JalonRojas2019; @Delandmeter2019]. In many cases, these packages are specific to particular particle classes or hydrodynamic models, or are written and embedded in proprietary software languages, and can be inflexible or difficult to integrate into different applications. In the case of plastic dispersal simulations, the underlying physical processes are still being researched and their implementation is under development [@vanSebille2020]. Hence, an open-source, flexible, and modular approach to performing Lagrangian simulations is necessary for prototyping, developing, and testing new physical process parameterisation schemes. Easy-to-run simulations allow for a more reproducable results, and for simple-to-produce sensitivity analyses.

Here, we have developed `plasticparcels` as an extension of the `parcels` Lagrangian framework [@Lange2017; @Delandmeter2019] in order to unify plastic dispersion modelling into one easy-to-use code. While `plasticparcels` has been designed for researchers who routinely perform plastic particle dispersion simulations, it is equally useful to novice users who are new to Lagrangian ocean analysis techniques. The core functionality of `parcels` are its `FieldSet`, `ParticleSet`, and `Kernel` objects. In the context of `plasticparcels`, the `FieldSet` contains a collection of hydrodynamic, physical, and biogeochemical fields required for plastic particle advection. The `ParticleSet` contains a set of plastic particles with their time-evolving properties, such as their position, accumulated biofilm amount, and density. A `Kernel` defines the specific computational behaviour for simulating a physical process applied to a plastic particle.

In `parcels`, these `Kernel` objects are designed to be as flexible and customisable as possible so that users can perform Lagrangian simulations of a wide variety of particulates, such as tuna, plastic, plankton, icebergs, turtles [@Lange2017]. However, due to the flexible nature of the software in general, there is a steep learning curve for new users, who often find it difficult to setup their simulations in a rapid fashion due to the complexity of modern hydrodynamic model output. We have developed `plasticparcels` as user-friendly tool specifically designed for easy-to-generate plastic dispersal simulations. While `plasticparcels` is primarily designed for use in the cloud and in HPC environments (due to the typically terabyte-size of hydrodynamic datasets generated from ocean general circulation models), it can be easily installed and run on local machines. A schematic of `plasticparcels` is shown in Fig. \ref{fig:schematic}.

![`plasticparcels` schematic.\label{fig:schematic}](schematic.png){width=100%}

# Usage Example
Here, we briefly demonstrate how `plasticparcels` can be used for a microplastic dispersal simulation in the Mediterranean Sea. The tutorial can be found on the [`plasticparcels` documentation](https://plastic.parcels-code.org/en/latest/examples/example_Italy_coast.html). Here, we use the coastal mismanaged plastic waste dataset [@Jambeck2015] to visualise the trajectories of buoyant (surface-bound) microplastic particles subject to the effects of Stokes drift and wind-induced drift, neglecting any vertical motion (along with any biofouling, or vertical mixing).

We start by importing `plasticparcels`, loading pre-defined settings that include information about the models and parameter settings for our simulation, and defining the start date, runtime, simulation timestep and output timestep.

```python
from datetime import datetime, timedelta
import plasticparcels as pp
settings_file = 'docs/examples/example_Italy_coast_settings.json'
settings = pp.utils.load_settings(settings_file)

settings['simulation'] = {
    'startdate': datetime.strptime('2019-01-01-00:00:00', '%Y-%m-%d-%H:%M:%S'),
    'runtime': timedelta(days=30),        # Runtime of simulation
    'outputdt': timedelta(hours=12),      # Timestep of output
    'dt': timedelta(minutes=20),          # Timestep of advection
    }
```

After turning on/off certain behaviour kernels, we define our plastic particle type, and particle release settings.

```python
settings['use_3D'] = False          # Turn off 3D advection
settings['use_biofouling'] = False  # Turn off biofouling
settings['use_stokes'] = True       # Turn on Stokes drift
settings['use_wind'] = True         # Turn on wind-induced drift

settings['plastictype'] = {
    'wind_coefficient' : 0.01,  # Percentage of wind to apply to particles
    'plastic_diameter' : 0.001, # Plastic particle diameter (m)
    'plastic_density' : 1030.,  # Plastic particle density (kg/m^3)
}

settings['release'] = {
    'initialisation_type': 'coastal',
    'country': 'Italy',
}
```

We then create the `FieldSet`, `ParticleSet` and `Kernel` list objects, and run our simulation.

```python
fieldset = pp.constructors.create_fieldset(settings)
pset = pp.constructors.create_particleset_from_map(fieldset, settings)
kernels = pp.constructors.create_kernel(fieldset)

pfile = pp.ParticleFile(
  'example_Italy_coast.zarr',
  pset,
  settings=settings,
  outputdt=settings['simulation']['outputdt']
)
pset.execute(
  kernels,
  runtime=settings['simulation']['runtime'],
  dt=settings['simulation']['dt'],
  output_file=pfile
)
```

A trajectory plot of the simulated plastic particles is shown in Fig. \ref{fig:example} a), and a concentration plot is shown in Fig. \ref{fig:example} b).

![The results of a simple simulation to identify the pathways of coastal mismanaged plastic waste [@Jambeck2015] along Italian coastlines. a) Trajectories of simulated plastic particles, b) The number of particles that enter a $0.1^\circ \times 0.1^\circ$ grid cell over the duration of the simulation.\label{fig:example}](example.png){width=100%}

# Acknowledgments
We would like to thank the Lagrangian Ocean Analysis team at Utrecht University for their helpful suggestions developing the tool. MCD was supported by the NECCTON project, which has received funding from Horizon Europe RIA under grant agreement No 101081273. EvS was supported by the project Tracing Marine Macroplastics by Unraveling the Ocean’s Multiscale Transport Processes with project number VI.C.222.025 of the research programme Talent Programme Vici 2022 which is financed by the Dutch Research Council (NWO).

# References