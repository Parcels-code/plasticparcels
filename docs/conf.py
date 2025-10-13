# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import inspect
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath('..')) # Allow to find `plasticparcels` package

import plasticparcels

project = 'plasticparcels'
copyright = f'{datetime.datetime.now().year}, The Parcels Team'
author = 'The Parcels Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

linkcheck_ignore = [
    r'http://localhost:\d+/',
    r"http://www2\.cesm\.ucar\.edu/models/cesm1\.0/pop2/doc/sci/POPRefManual.pdf",  # Site doesn't allow crawling
    r"https://pubs\.acs\.org/doi/10\.1021/acs\.est\.0c01984",  # Site doesn't allow crawling
    r"https://aip\.scitation\.org/doi/10\.1063/1\.4982720",  # Site doesn't allow crawling
    r"https://www\.sciencedirect\.com/.*",  # Site doesn't allow crawling
    r"https://lxml\.de/",  # Crawler occasionally fails to establish connection
    r"https://linux\.die\.net/",  # Site doesn't allow crawling
    r"https://agupubs\.onlinelibrary\.wiley\.com/",  # Site doesn't allow crawling

    # To monitor
    r"http://marine.copernicus.eu/",  # 2023-06-07 Site non-responsive
    r"https://www\.nodc\.noaa\.gov/",  # 2023-06-23 Site non-responsive
    r"https://mybinder\.org/",  # 2023-09-02 Site non-responsive
    r"https://ariane-code.cnrs.fr/",  # 2024-04-30 Site non-responsive
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "logo": {
        "image_light": "plasticparcelslogo.png",
        "image_dark": "plasticparcelslogo_inverted.png",
    },
    # "use_edit_page_button": True,
    "github_url": "https://github.com/Parcels-code/plasticparcels",
    "icon_links": [
        {
            "name": "Conda Forge",
            "url": "https://anaconda.org/conda-forge/plasticparcels",  # required
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        }
    ]
}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "favicon.ico"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
    "numpydoc",
    'sphinx.ext.autosummary',
]

myst_enable_extensions = ["dollarmath", "amsmath"]

def is_non_standard_version(version):
    """
    Determine if the given version string has more than 3 components (e.g., from a developer version).
    """

    return len(version.split(".")) != 3

# based on pandas doc/source/conf.py
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(plasticparcels.__file__))

    if is_non_standard_version(plasticparcels.__version__):
        return f"https://github.com/Parcels-code/plasticparcels/blob/main/plasticparcels/{fn}{linespec}"
    else:
        return (
            f"https://github.com/Parcels-code/plasticparcels/blob/"
            f"v{plasticparcels.__version__}/plasticparcels/{fn}{linespec}"
        )
