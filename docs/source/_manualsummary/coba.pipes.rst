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

Readers
~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst
      
      CsvReader
      ArffReader
      LibsvmReader
      ManikReader
      
Sources
~~~~~~~~~~~~~~~

   .. autosummary::
      :toctree: ../_autosummary
      :template: class_with_ctor.rst
            
      ListSource
      DiskSource
      HttpSource
      QueueSource
      
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
