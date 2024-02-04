.. _coba-primitives:

coba.primitives
===============

.. automodule:: coba.primitives

Type Aliases
~~~~~~~~~~~~

These exist to show how data and variables flow through coba objects.

   .. autosummary::

      Context
      Action
      Actions
      Reward
      Prob
      Kwargs
      Pred

Interfaces
~~~~~~~~~~
   .. autosummary::
      :toctree:

      Environment
      Learner
      Evaluator
      Rewards
      EnvironmentFilter

Rewards
~~~~~~~
   .. autosummary::
      :toctree:

      L1Reward
      BinaryReward
      HammingReward
      DiscreteReward

Interactions
~~~~~~~~~~~~
   .. autosummary::
      :toctree:

      Interaction
      SimulatedInteraction
      LoggedInteraction
      GroundedInteraction
