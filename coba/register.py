
from coba.registry    import CobaRegistry
from coba.pipes       import NullIO, ConsoleIO, DiskIO
from coba.config      import DiskCacher, NoneCacher, MemoryCacher, IndentLogger, NoneLogger, BasicLogger
from coba.simulations import OpenmlSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation
from coba.simulations import Identity, Shuffle, Take, Sort, Scale, Cycle
from coba.experiments import BenchmarkFileFmtV2

CobaRegistry.register("NoneSink"   , NullIO   )
CobaRegistry.register("DiskSink"   , DiskIO   )
CobaRegistry.register("ConsoleSink", ConsoleIO)

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

CobaRegistry.register("Identity", Identity)
CobaRegistry.register("Take"    , Take    )
CobaRegistry.register("Shuffle" , Shuffle )
CobaRegistry.register("Sort"    , Sort    )
CobaRegistry.register("Scale"   , Scale   )
CobaRegistry.register("Cycle"   , Cycle   )

CobaRegistry.register("BenchmarkFileV2", BenchmarkFileFmtV2)