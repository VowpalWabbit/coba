{# This template creates a class stub with all attributes and methods in a single file #}

{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members: __init__, __new__, mro

   {% block constructor %}
   {% if "__init__" not in inherited_members %}
   .. rubric:: {{ _('Constructors') }}

   .. automethod:: __init__
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% set clean_methods = methods | reject("in",["__init__","mro"]) | list %}
   {% if clean_methods %}
   .. rubric:: {{ _('Methods') }}

   {% for item in clean_methods %}
   .. automethod:: {{item}}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{item}}
   {%- endfor %}
   {% endif %}
   {% endblock %}
