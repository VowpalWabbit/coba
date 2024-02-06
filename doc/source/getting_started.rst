===============
Getting Started
===============

``Coba`` is a Python package supporting algorithmic and applied contextual bandit research.

Installation
~~~~~~~~~~~~

 ``Coba`` can be installed via pip.

.. code-block:: bash

   $ pip install coba

Dependencies
~~~~~~~~~~~~

``Coba`` has no hard dependencies, but it does have optional depdencies for certain functionality.

The examples contained in this documentation use the following optional dependencies.

.. code-block:: bash

   $ pip install matplotlib pandas scipy numpy vowpalwabbit

Key Concepts
~~~~~~~~~~~~

.. note::
   ``Coba`` is organized around six key concepts:

   1. Interaction -- A single decision point (i.e. a context with actions and rewards).
   2. Environment -- A sequence of Interactions
   3. Learner -- An algorithm for learning and acting in an Environment
   4. Evaluator -- A method to evaluate a Learner in an Environment
   5. Experiment -- A collection of Environments, Learners, and Evaluators
   6. Result -- Performance data produced by an Experiment

   Knowing these concepts can help you find help and perform advanced experiments.

The core concepts help in finding more information about ``Coba``. For example, all built-in learners can be
found at :ref:`coba-learners`. Help with creating environments can be found at :ref:`coba-environments`. The types of evaluation
that coba supports out of the box can be found at :ref:`coba-evaluators`. The various ways an experiment can be configured is
described at :ref:`coba-experiments`. And details regarding analysis functionality can be found at :ref:`coba-results`.

Next Steps
~~~~~~~~~~

 * For *all* researchers we suggest reading about how to create your :doc:`first experiment <notebooks/First_Experiment>`.
 * For *algorithm* researchers we suggest reading about how to create :doc:`custom learners <notebooks/Learners>`.
 * For *applied* researchers we suggest reading about creating :doc:`custom environments <notebooks/Environments>`.
