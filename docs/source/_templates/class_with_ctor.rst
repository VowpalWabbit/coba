{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members: __init__, __new__
   :show-inheritance:

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
      
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :template: base.rst
   {% for item in methods %}
      {% if item != "__init__" %}
         ~{{ name }}.{{ item }}
      {% endif %}
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
