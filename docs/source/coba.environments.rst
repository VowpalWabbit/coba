.. _coba-environments:

coba.environments
=================

.. automodule:: coba.environments

Core
~~~~

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst

      Environments

Interfaces
~~~~~~~~~~

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst

      Environment
      SimulatedEnvironment
      LoggedEnvironment
      WarmStartEnvironment

Interaction Types
~~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst

      Interaction
      SimulatedInteraction
      LoggedInteraction

Simulated Environments
~~~~~~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst

      MemorySimulation
      LambdaSimulation
      ClassificationSimulation
      RegressionSimulation
      OpenmlSimulation
      CsvSimulation
      ArffSimulation
      LibsvmSimulation
      ManikSimulation
      LinearSyntheticSimulation
      LocalSyntheticSimulation
      
Environment Filters
~~~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst      

      Sort
      Scale
      Cycle
      Impute
      Binary
      WarmStart      
      Shuffle 
      Take
      Reservoir
      Identity
      Sparse
      FilteredEnvironment
      EnvironmentFilter
