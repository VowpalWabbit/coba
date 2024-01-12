Result
======

.. currentmodule:: coba.results

.. autoclass:: Result
   :exclude-members: __init__, __new__, mro

   .. rubric:: Constructors
   .. automethod:: __init__

   .. rubric:: Methods
   .. automethod:: copy
   .. automethod:: filter_best
   .. automethod:: filter_env
   .. automethod:: filter_fin
   .. automethod:: filter_int
   .. automethod:: filter_lrn
   .. automethod:: filter_val
   .. automethod:: from_file
   .. automethod:: from_logged_envs
   .. automethod:: from_save
   .. automethod:: from_source
   .. automethod:: plot_contrast
   .. automethod:: plot_learners
   .. automethod:: raw_contrast
   .. automethod:: raw_learners
   .. automethod:: set_plotter
   .. automethod:: where

   .. rubric:: Attributes
   .. autoattribute:: environments
   .. autoattribute:: evaluators
   .. autoattribute:: interactions
   .. autoattribute:: learners
