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
exclude_patterns = []


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
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Options for autodoc ----------------------------------------------------

autodoc_docstring_signature = True
autoclass_content = "class"
autodoc_class_signature = "separated"
#autodoc_member_order = "bysource"

# This gives coba a consistent public interface in the documentation.
# An alternative way to do this might be :canonical: on ..py:class::
# https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#directive-py-class
import coba.learners
import coba.environments
import coba.experiments
import coba.pipes
import coba.contexts
import coba.contexts.core

def set_module(module):
    for cls in map(module.__dict__.get, module.__all__):
        try:
            cls.__module__ = module.__name__
        except:
            pass

set_module(coba.learners)
set_module(coba.environments)
set_module(coba.experiments)
set_module(coba.pipes)
set_module(coba.contexts)

#we have to point to meta because sphinx can't handle class level properties
coba.contexts.core.CobaContext_meta.__module__ = "coba.contexts"
coba.contexts.core.CobaContext_meta.__name__ = "CobaContext"
coba.contexts.__dict__['CobaContext'] = coba.contexts.core.CobaContext_meta 

#we have to point to meta because sphinx can't handle class level properties
coba.contexts.core.LearnerContext_meta.__module__ = "coba.contexts"
coba.contexts.core.LearnerContext_meta.__name__ = "LearnerContext"
coba.contexts.__dict__['LearnerContext'] = coba.contexts.core.LearnerContext_meta

autosummary_generate_overwrite = False
