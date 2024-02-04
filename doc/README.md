The directive at the top of files are reference labels
    * https://www.sphinx-doc.org/en/master/usage/referencing.html#role-ref

We make use of two extensions, autodoc and autosummary:
    * Autodoc creates pages with docstrings from classes, members, and attributes
    * Autosummary makes a table of elements with short descriptions

The left menu is created by our base theme and the table-of-contents in index.rst.
    * https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html#how-the-table-of-contents-displays

When adding a new top level module we can use autosummary to generate stubs. To do this set a template:
   .. autosummary::
      :toctree:
      :template: class_with_ctor.rst
      CobaContext

To build website locally run:
    * sphinx-build -M html docs/source docs/out