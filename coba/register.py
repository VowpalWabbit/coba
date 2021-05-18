from coba.simulations.core import ClassificationSimulation
from coba.data.filters     import ArffReader, CsvReader, LibSvmReader
from coba.data.sources     import DiskSource, HttpSource
from coba.data.sinks       import NoneSink, ConsoleSink, DiskSink
from coba.tools            import CobaRegistry, DiskCacher, NoneCacher, MemoryCacher, IndentLogger, NoneLogger, BasicLogger
from coba.simulations      import OpenmlSimulation, PCA, Shuffle, Take, Sort
from coba.benchmarks       import BenchmarkFileFmtV1, BenchmarkFileFmtV2

CobaRegistry.register("NoneSink"   , NoneSink   )
CobaRegistry.register("DiskSink"   , DiskSink   )
CobaRegistry.register("ConsoleSink", ConsoleSink)

CobaRegistry.register("BasicLogger" , BasicLogger )
CobaRegistry.register("IndentLogger", IndentLogger)
CobaRegistry.register("NoneLogger"  , NoneLogger  )

CobaRegistry.register("DiskCacher"   , DiskCacher  )
CobaRegistry.register("NoneCacher"   , NoneCacher  )
CobaRegistry.register("MemoryCacher" , MemoryCacher)

CobaRegistry.register("DiskSource", DiskSource)
CobaRegistry.register("HttpSource", HttpSource)

CobaRegistry.register("CsvReader"   , CsvReader   )
CobaRegistry.register("ArffReader"  , ArffReader  )
CobaRegistry.register("LibSvmReader", LibSvmReader)

CobaRegistry.register("OpenmlSimulation"        , OpenmlSimulation        )
CobaRegistry.register("ClassificationSimulation", ClassificationSimulation)

CobaRegistry.register("Take"            , Take   )
CobaRegistry.register("Shuffle"         , Shuffle)
CobaRegistry.register("Sort"            , Sort   )
CobaRegistry.register("PCA"             , PCA    )

CobaRegistry.register("BenchmarkFileV1", BenchmarkFileFmtV1)
CobaRegistry.register("BenchmarkFileV2", BenchmarkFileFmtV2)