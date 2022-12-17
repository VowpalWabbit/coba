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

Interfaces
~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Environment
      SimulatedEnvironment
      LoggedEnvironment

Interaction Types
~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Interaction
      SimulatedInteraction
      LoggedInteraction

Raw Data Sources
~~~~~~~~~~~~~~~~
   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      CsvSource
      ArffSource
      LibsvmSource
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

Environment Filters
~~~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      EnvironmentFilter
      Sort
      Scale
      Cycle
      Impute
      Binary
      Shuffle
      Take
      Reservoir
      Identity
      Sparse
      Where
