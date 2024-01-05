.. _coba-environments-Environments:

Environments
============

.. currentmodule:: coba.environments

.. autoclass:: Environments

   .. rubric:: Settings

   .. autosummary::
      :toctree: ../_autosummary
      :template: base.rst

      ~Environments.cache_dir

   .. rubric:: Creators

   .. autosummary::
      :toctree: ../_autosummary
      :template: base.rst

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

   .. rubric:: Filters

   .. autosummary::
      :toctree: ../_autosummary
      :template: base.rst

      ~Environments.batch
      ~Environments.binary
      ~Environments.cache
      ~Environments.chunk
      ~Environments.count
      ~Environments.cycle
      ~Environments.dense
      ~Environments.filter
      ~Environments.flatten
      ~Environments.grounded
      ~Environments.impute
      ~Environments.index
      ~Environments.logged
      ~Environments.materialize
      ~Environments.noise
      ~Environments.ope_rewards
      ~Environments.params
      ~Environments.repr
      ~Environments.reservoir
      ~Environments.riffle
      ~Environments.save
      ~Environments.scale
      ~Environments.shuffle
      ~Environments.slice
      ~Environments.sort
      ~Environments.sparse
      ~Environments.take
      ~Environments.unbatch
      ~Environments.where
