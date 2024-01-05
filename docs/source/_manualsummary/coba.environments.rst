.. _coba-environments:

coba.environments
=================

.. automodule:: coba.environments

   .. rubric:: Core

   .. autosummary::
      :toctree: ../_manualsummary
      :template: class_with_ctor.rst

      Environments

   .. rubric:: Raw Data Sources

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      CsvSource
      ArffSource
      LibSvmSource
      ManikSource
      OpenmlSource

   .. rubric:: Simulated Environments

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      LambdaSimulation
      SupervisedSimulation
      OpenmlSimulation
      LinearSyntheticSimulation
      NeighborsSyntheticSimulation
      KernelSyntheticSimulation
      MLPSyntheticSimulation

   .. rubric:: Environment Filters

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Sort
      Scale
      Cycle
      Impute
      Binary
      Shuffle
      Take
      Reservoir
      Identity
      Sparsify
      Densify
      Where
