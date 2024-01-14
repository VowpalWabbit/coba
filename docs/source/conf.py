# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0,os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'coba'
copyright = '2021, Mark Rucker'
author = 'Mark Rucker'

# The full version, including alpha/beta/rc tags
release = '4.5'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [ ]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# To build with this theme locally you must install it first via `pip install sphinx-rtd-theme`
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [ "_statics" ]

html_theme_options = {
    #'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 2,
    'includehidden': True,
    'titles_only': False
}

# -- Options for autodoc ----------------------------------------------------

autodoc_class_signature = "separated"
autodoc_type_aliases = {
    'Context': 'Context',
    'Action' : 'Action',
    'Actions': 'Actions',
    'Reward' : 'Reward',
    'Prob'   : 'Prob',
    'Pred'   : 'Pred',
    'Kwargs' : 'Kwargs'
}

# -- Options for summary ----------------------------------------------------

autosummary_generate_overwrite = False

# This gives coba a consistent public interface in the documentation.
# An alternative way to do this might be :canonical: on ..py:class::
# https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#directive-py-class
import coba.learners
import coba.environments
import coba.experiments
import coba.evaluators
import coba.results
import coba.context
import coba.primitives
import coba.context.core

def set_module(module):
    from types import ModuleType
    for cls in [v for k,v in module.__dict__.items() if not k.startswith("__") and not isinstance(v,ModuleType) ]:
        try:
            cls.__module__ = module.__name__
        except:
            pass

set_module(coba.learners)
set_module(coba.environments)
set_module(coba.experiments)
set_module(coba.context)

#we have to point to meta because sphinx can't handle class level properties
coba.context.core.CobaContext_meta.__module__ = "coba.context"
coba.context.core.CobaContext_meta.__name__ = "CobaContext"
coba.context.__dict__['CobaContext'] = coba.context.core.CobaContext_meta
