{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members: __init__, __new__, mro

   {% block constructor %}
   {% if "__init__" not in inherited_members %}

   .. rubric:: {{ _('Constructors') }}

   .. autosummary::
      :toctree:
      :template: base.rst
   {% for item in methods %}
   {% if item == "__init__" %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endfor %}

   {% endif %}
   {% endblock %}
   

   {% block methods %}
      
   {% set clean_methods = methods | reject("in",["__init__","mro"]) | list %}
   {% if clean_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :template: base.rst
   {% for item in clean_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
      :template: base.rst
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
