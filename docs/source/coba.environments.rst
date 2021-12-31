coba.environments
=================

.. automodule:: coba.environments

   .. rubric:: Core

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst
      
      Environments

   .. rubric:: Interfaces

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst
      
      Environment
      SimulatedEnvironment
      LoggedEnvironment
      WarmStartEnvironment

   .. rubric:: Interaction Types

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst

      Interaction
      SimulatedInteraction
      LoggedInteraction

   .. rubric:: Simulated Environments

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
      
   .. rubric:: Environment Filters

   .. autosummary::
      :toctree: _autosummary
      :template: class_with_ctor.rst      
      
      EnvironmentFilter
      FilteredEnvironment
      Sort
      Scale
      Cycle
      Impute
      Binary
      ToWarmStart      
      Shuffle 
      Take
      Reservoir
      Identity
      Sparse
