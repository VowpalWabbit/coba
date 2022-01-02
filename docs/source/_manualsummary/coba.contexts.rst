.. _coba-contexts:

coba.contexts
=============

.. automodule:: coba.contexts

Core
~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      CobaContext
      LearnerContext

Interfaces
~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Cacher
      Logger

Cachers
~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst
      
      NullCacher
      MemoryCacher
      DiskCacher
      ConcurrentCacher
      
Loggers
~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst
      
      NullLogger
      BasicLogger
      IndentLogger
      DecoratedLogger

Log Decorators
~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst
    
      ExceptLog
      NameLog
      StampLog
