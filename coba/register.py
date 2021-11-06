
from coba.registry     import CobaRegistry
from coba.pipes        import NullIO, ConsoleIO, DiskIO
from coba.environments import OpenmlSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation
from coba.environments import Identity, Shuffle, Take, Sort, Scale, Cycle
from coba.config       import DiskCacher, NullCacher, MemoryCacher, IndentLogger, NullLogger, BasicLogger

CobaRegistry.register("Null"   , NullIO   )
CobaRegistry.register("Disk"   , DiskIO   )
CobaRegistry.register("Console", ConsoleIO)

CobaRegistry.register("BasicLogger" , BasicLogger )
CobaRegistry.register("IndentLogger", IndentLogger)
CobaRegistry.register("NullLogger"  , NullLogger  )

CobaRegistry.register("DiskCacher"   , DiskCacher  )
CobaRegistry.register("NullCacher"   , NullCacher  )
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