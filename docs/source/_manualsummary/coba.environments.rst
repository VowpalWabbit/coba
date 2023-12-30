.. _coba-environments:

coba.environments
=================

.. automodule:: coba.environments

Core
~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Environments

Raw Data Sources
~~~~~~~~~~~~~~~~
   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      CsvSource
      ArffSource
      LibSvmSource
      ManikSource
      OpenmlSource


Simulated Environments
~~~~~~~~~~~~~~~~~~~~~~

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

Environment Filters
~~~~~~~~~~~~~~~~~~~

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
