
from coba.registry     import CobaRegistry
from coba.pipes        import NullIO, ConsoleIO, DiskIO, ListIO
from coba.environments import OpenmlSimulation
from coba.environments import Sort, Scale, Cycle, Shuffle, Take, Identity
from coba.contexts     import DiskCacher, NullCacher, MemoryCacher, IndentLogger, NullLogger, BasicLogger

CobaRegistry.register("Null"   , NullIO   )
CobaRegistry.register("Disk"   , DiskIO   )
CobaRegistry.register("Memory" , ListIO )
CobaRegistry.register("Console", ConsoleIO)

CobaRegistry.register("BasicLogger" , BasicLogger )
CobaRegistry.register("IndentLogger", IndentLogger)
CobaRegistry.register("NullLogger"  , NullLogger  )

CobaRegistry.register("DiskCacher"   , DiskCacher  )
CobaRegistry.register("NullCacher"   , NullCacher  )
CobaRegistry.register("MemoryCacher" , MemoryCacher)

CobaRegistry.register("OpenmlSimulation", OpenmlSimulation)

CobaRegistry.register("Identity", Identity)
CobaRegistry.register("Take"    , Take    )
CobaRegistry.register("Shuffle" , Shuffle )
CobaRegistry.register("Sort"    , Sort    )
CobaRegistry.register("Scale"   , Scale   )
CobaRegistry.register("Cycle"   , Cycle   )