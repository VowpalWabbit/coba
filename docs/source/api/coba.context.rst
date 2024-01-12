.. _coba-context:

coba.context
============

.. automodule:: coba.context

Core
~~~~
   .. autosummary::
      :toctree:

      CobaContext

Interfaces
~~~~~~~~~~
   .. autosummary::
      :toctree:

      Cacher
      Logger

Cachers
~~~~~~~
   .. autosummary::
      :toctree:

      NullCacher
      MemoryCacher
      DiskCacher
      ConcurrentCacher

Loggers
~~~~~~~
   .. autosummary::
      :toctree:

      NullLogger
      BasicLogger
      IndentLogger
      DecoratedLogger

Log Decorators
~~~~~~~~~~~~~~
   .. autosummary::
      :toctree:

      ExceptLog
      NameLog
      StampLog
