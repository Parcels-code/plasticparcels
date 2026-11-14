.. plasticparcels documentation master file, created by
   sphinx-quickstart on Fri May  3 09:55:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Plasticparcels documentation
============================

Welcome to the documentation of ``plasticparcels``.

``plasticparcels`` is a python package for simulating the transport and dispersion of plastics in the ocean. The tool is based on the ``parcels`` computational Lagrangian ocean analysis framework, providing a modular and customisable collection of methods, notebooks, and tutorials for advecting virtual plastic particles with a wide range of physical properties.

The manuscript detailing the first release of ``plasticparcels``, version 0.3.1, has been published in the `Journal of Open Source Software (JOSS) <https://doi.org/10.21105/joss.07094>`_.

Installation
^^^^^^^^^^^^

``plasticparcels`` can be installed using ``conda`` from the `conda-forge channel <https://anaconda.org/conda-forge/plasticparcels>`_ with the following command:

.. code-block::

   conda install conda-forge::plasticparcels


Or downloaded from https://github.com/Parcels-code/plasticparcels


Required Data
^^^^^^^^^^^^^

``plasticparcels`` has been developed for use with data from the Copernicus Marine Service, and requires a `free account <https://marine.copernicus.eu/register-copernicus-marine-service>`_.

Just like the ``parcels`` framework, ``plasticparcels`` can be adapted to use other hydrodynamic, biogeochemical, wave, and atmospheric models. If you require assistance, please contact us through the `Discussions page on GitHub <https://github.com/Parcels-code/plasticparcels/discussions>`_.

Community Guidelines
^^^^^^^^^^^^^^^^^^^^

We welcome contributions to ``plasticparcels``, especially example workbooks and analyses for our `public examples page <https://plastic.parcels-code.org/en/latest/examples.html>`_. To contribute to the project, please submit a `pull request <https://github.com/Parcels-code/plasticparcels/pulls>`_.

If you want to request a new feature, or if you find an issue or bug in the code, please open an issue in the `plasticparcels issue tracker <https://github.com/Parcels-code/plasticparcels/issues>`_.

If you would like support using ``plasticparcels``, or are have any questions about your ``plasticparcels`` simulations, please start a discussion in the `plasticparcels discussion page <https://github.com/Parcels-code/plasticparcels/discussions>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Home <self>
   Examples <examples>
   Physics kernels <physicskernels>
   Plastic initialisation maps <initialisationmaps>
   API reference <reference/index>
   Parcels website <https://parcels-code.org/>
