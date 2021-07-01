
from coba.registry    import CobaRegistry
from coba.pipes       import NoneSink, ConsoleSink, DiskSink
from coba.config      import DiskCacher, NoneCacher, MemoryCacher, IndentLogger, NoneLogger, BasicLogger
from coba.simulations import OpenmlSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation, PCA, Shuffle, Take, Sort
from coba.benchmarks  import BenchmarkFileFmtV2

CobaRegistry.register("NoneSink"   , NoneSink   )
CobaRegistry.register("DiskSink"   , DiskSink   )
CobaRegistry.register("ConsoleSink", ConsoleSink)

CobaRegistry.register("BasicLogger" , BasicLogger )
CobaRegistry.register("IndentLogger", IndentLogger)
CobaRegistry.register("NoneLogger"  , NoneLogger  )

CobaRegistry.register("DiskCacher"   , DiskCacher  )
CobaRegistry.register("NoneCacher"   , NoneCacher  )
CobaRegistry.register("MemoryCacher" , MemoryCacher)

CobaRegistry.register("OpenmlSimulation", OpenmlSimulation)
CobaRegistry.register("LibsvmSimulation", LibsvmSimulation)
CobaRegistry.register("CsvSimulation"   , CsvSimulation   )
CobaRegistry.register("ArffSimulation"  , ArffSimulation  )

CobaRegistry.register("Take"            , Take   )
CobaRegistry.register("Shuffle"         , Shuffle)
CobaRegistry.register("Sort"            , Sort   )
CobaRegistry.register("PCA"             , PCA    )

CobaRegistry.register("BenchmarkFileV2", BenchmarkFileFmtV2)