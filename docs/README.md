The directives at the top of files are reference labels
https://www.sphinx-doc.org/en/master/usage/referencing.html#role-ref

We make use of two extensions, autodoc and autosummary:
    * Autodoc allows us to make documentation pages with docstrings from classes, members, and attributes
    * Autosummary makes lists of pages from a template file and autodoc

The left menu is created by our base theme and the table-of-contents in index.rst.
https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html#how-the-table-of-contents-displays

Using autosummary we generated a stub for every class. These stubs will auto-update their docstrings when the Python code changes. What won't auto-update are new methods and attributes because we manually define each class.