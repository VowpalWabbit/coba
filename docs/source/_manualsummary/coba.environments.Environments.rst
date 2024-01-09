.. _coba-environments-Environments:

Environments
============

.. currentmodule:: coba.environments

.. autoclass:: Environments
   :exclude-members: __init__, __new__, mro

   .. rubric:: Create
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.from_dataframe
      ~Environments.from_custom
      ~Environments.from_feurer
      ~Environments.from_openml
      ~Environments.from_linear_synthetic
      ~Environments.from_kernel_synthetic
      ~Environments.from_neighbors_synthetic
      ~Environments.from_mlp_synthetic
      ~Environments.from_result
      ~Environments.from_save
      ~Environments.from_supervised
      ~Environments.from_template

   .. rubric:: Select
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.slice
      ~Environments.take
      ~Environments.reservoir
      ~Environments.where

   .. rubric:: Reorder
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.riffle
      ~Environments.shuffle
      ~Environments.sort

   .. rubric:: Precondition
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.dense
      ~Environments.flatten
      ~Environments.impute
      ~Environments.repr
      ~Environments.scale
      ~Environments.sparse

   .. rubric:: Noise
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.cycle
      ~Environments.noise

   .. rubric:: Transform
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.binary
      ~Environments.grounded
      ~Environments.logged
      ~Environments.ope_rewards

   .. rubric:: Control
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.batch
      ~Environments.cache
      ~Environments.chunk
      ~Environments.materialize
      ~Environments.save
      ~Environments.unbatch

   .. rubric:: Other
   .. autosummary::
      :toctree: ../_autosummary
      :nosignatures:

      ~Environments.filter
      ~Environments.cache_dir
      ~Environments.params
