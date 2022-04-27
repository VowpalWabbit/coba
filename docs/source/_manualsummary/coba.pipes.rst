coba.pipes
=============

.. automodule:: coba.pipes

Interfaces
~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Source
      Filter
      Sink

Core
~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Pipes

Execution Pipes
~~~~~~~~~~~~~~~~~
   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Foreach
      Multiprocessor


Sources
~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      ListSource
      DiskSource
      HttpSource
      QueueSource
      UrlSource


Reader Filters
~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      CsvReader
      ArffReader
      LibsvmReader
      ManikReader

Table Filters
~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Structure
      Flatten
      Default
      Encode

Utility Filters
~~~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      Take
      Shuffle
      Identity
      Default
      Reservoir
      JsonDecode
      JsonEncode

Sinks
~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst

      NullSink
      ListSink
      DiskSink
      ConsoleSink
      QueueSink
