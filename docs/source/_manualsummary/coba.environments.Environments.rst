Environments
============

.. currentmodule:: coba.environments

.. autoclass:: Environments
   :exclude-members: __init__, __new__, mro

   .. rubric:: Static Constructors

   .. automethod:: from_custom
   .. automethod:: from_dataframe
   .. automethod:: from_feurer
   .. automethod:: from_kernel_synthetic
   .. automethod:: from_linear_synthetic
   .. automethod:: from_mlp_synthetic
   .. automethod:: from_neighbors_synthetic
   .. automethod:: from_openml
   .. automethod:: from_result
   .. automethod:: from_save
   .. automethod:: from_supervised
   .. automethod:: from_template

   .. rubric:: Methods

   .. automethod:: batch
   .. automethod:: binary
   .. automethod:: cache
   .. automethod:: cache_dir
   .. automethod:: chunk
   .. automethod:: cycle
   .. automethod:: dense
   .. automethod:: filter
   .. automethod:: flatten
   .. automethod:: grounded
   .. automethod:: impute
   .. automethod:: logged
   .. automethod:: materialize
   .. automethod:: noise
   .. automethod:: ope_rewards
   .. automethod:: params
   .. automethod:: repr
   .. automethod:: reservoir
   .. automethod:: riffle
   .. automethod:: save
   .. automethod:: scale
   .. automethod:: shuffle
   .. automethod:: slice
   .. automethod:: sort
   .. automethod:: sparse
   .. automethod:: take
   .. automethod:: unbatch
   .. automethod:: where