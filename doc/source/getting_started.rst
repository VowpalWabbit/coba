===============
Getting Started
===============

Coba is a Python package supporting algorithmic and applied contextual bandit research.

Installation
~~~~~~~~~~~~

Coba can be installed via pip.

.. code-block:: bash

   $ pip install coba

Coba has no hard dependencies, but it does have optional depdencies for certain functionality.

Coba will let you know if a dependency is needed, so there is no need to install things upfront.

The examples contained in the documentation use the following optional dependencies.

.. code-block:: bash

   $ pip install matplotlib pandas scipy numpy vowpalwabbit cloudpickle

Contextual Bandits
~~~~~~~~~~~~~~~~~~

A contextual bandit (sometimes called a contextual multi-armed bandit) is an abstract game where players
repeatedly interact with a "contextual bandit". In each interaction the contextual bandit presents the
player with a context and a choice of actions.

The player must choose an action to play from the set of presented actions. If the player chooses well 
the contextual bandit gives a large reward. If the player chooses poorly the contextual bandit gives a 
small reward. The player only observes the reward for the action they chose.

The player's goal is to earn as much reward as possible. To succeed players need to learn what actions
give large rewards in what contexts. This game is of interest to researchers because it ammenable to
mathematical analysis while also being applicable to many real world problems.

Contextual Bandit Learners
~~~~~~~~~~~~~~~~~~~~~~~~~~

Contextual bandit learners are machine learning algorithms that have been designed to play contextual bandits.
They are particularly good at solving problems with partial feedback. That is, problems where we can observe
the consequence of an action but don't entirely know what would have happened had we done something else.

About Coba
~~~~~~~~~~

Coba was built to make it easier to experiment with contextual bandits and contextual bandit learners. More specifically,
coba supports two use cases.

One, evaluating a contextual bandit learner on *many* contextual bandits. This is useful for algorithm researchers who want
to broadly evaluate a contextual bandit learner's capabilities. Coba achieves this by creating contextual bandits from the
hundreds of real-world supervised datasets hosted on openml.org.

Two, evaluating *many* contextual bandit learners on a particular contextual bandit. This is useful for application
researchers who are trying to solve a specific use-case. Coba achieves this by providing robust implementations of
well-known algorithms behind a common interface.

Key Concepts
~~~~~~~~~~~~

.. note::
   ``Coba`` is organized around six key concepts:

   1. Interaction -- A single interaction with a contextual bandit.
   2. Environment -- A sequence of interactions with a specific contextual bandit.
   3. Learner -- A player in a contextual bandit game.
   4. Evaluator -- A method to evaluate how well a Learner plays an Environment
   5. Experiment -- A collection of Environments, Learners, and Evaluators.
   6. Result -- Data generated when an Experiment Evaluates Learners in an Environment.

   Knowing these concepts can help you find help and perform advanced experiments.

Knowing the key concepts makes it easier to find help and information about Coba.
For example, notebooks for each of the six key concepts are available under Basic Examples on the left.
All built-in learners can be found at :ref:`coba-learners`.
All environments and their filters can be found at :ref:`coba-environments`.
The types of evaluators that coba supports out of the box can be found at :ref:`coba-evaluators`.
The various ways an experiment can be configured is described at :ref:`coba-experiments`.
And details regarding analysis functionality can be found at :ref:`coba-results`.

Next Steps
~~~~~~~~~~

 * For *algorithm* researchers we suggest looking at :doc:`First Algorithm <notebooks/First_Algorithm>`.
 * For *application* researchers we suggest looking at :doc:`First Application <notebooks/First_Application>`.
