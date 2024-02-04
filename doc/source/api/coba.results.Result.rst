Result
======

.. currentmodule:: coba.results

.. autoclass:: Result
   :exclude-members: __init__, __new__, mro

   .. rubric:: Static
   .. automethod:: from_save

   .. rubric:: Methods
   .. automethod:: where_best
   .. automethod:: where_fin
   .. automethod:: where
   .. automethod:: plot_contrast
   .. automethod:: plot_learners
   .. automethod:: raw_contrast
   .. automethod:: raw_learners

   .. rubric:: Attributes
   .. autoattribute:: environments
      :annotation: : Table
   .. autoattribute:: evaluators
      :annotation: : Table
   .. autoattribute:: interactions
      :annotation: : Table
   .. autoattribute:: learners
      :annotation: : Table
