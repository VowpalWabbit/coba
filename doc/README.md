The directive at the top of files are reference labels
    * https://www.sphinx-doc.org/en/master/usage/referencing.html#role-ref

We make use of two extensions, autodoc and autosummary:
    * Autodoc creates content/elements on rst pages from Python docstrings
    * Autosummary makes a table of elements with short descriptions. Autosummary can also
        auto-generate stub files. Once the stub files are generated we no longer need that
        functionality and only use Autosummary to create the tables.

The left menu is created by our base theme and the table-of-contents in index.rst.
    * https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html#how-the-table-of-contents-displays

When adding a new top level module we can use autosummary to generate stubs. To do this set a template:
   .. autosummary::
      :toctree:
      :template: class_with_ctor.rst
      CobaContext

To build website locally run:
    * sphinx-build -M html doc/source doc/out
