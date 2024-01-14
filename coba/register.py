from coba.registry     import CobaRegistry
from coba.pipes        import NullSink, ConsoleSink, DiskSink, HttpSource
from coba.environments import OpenmlSimulation, SupervisedSimulation
from coba.environments import Sort, Scale, Cycle, Shuffle, Take, Identity, Where, Repr, Reservoir
from coba.context      import DiskCacher, NullCacher, IndentLogger, NullLogger, BasicLogger
from coba.primitives   import L1Reward, HammingReward, BinaryReward, DiscreteReward

CobaRegistry.register("range" , range)
CobaRegistry.register("zip"   , zip  )

CobaRegistry.register("Null"   , NullSink   )
CobaRegistry.register("Disk"   , DiskSink   )
CobaRegistry.register("Http"   , HttpSource )
CobaRegistry.register("Console", ConsoleSink)

CobaRegistry.register("BasicLogger" , BasicLogger )
CobaRegistry.register("IndentLogger", IndentLogger)
CobaRegistry.register("NullLogger"  , NullLogger  )

CobaRegistry.register("DiskCacher", DiskCacher)
CobaRegistry.register("NullCacher", NullCacher)

CobaRegistry.register("OpenmlSimulation"    , OpenmlSimulation    )
CobaRegistry.register("SupervisedSimulation", SupervisedSimulation)

CobaRegistry.register("Identity" , Identity )
CobaRegistry.register("Take"     , Take     )
CobaRegistry.register("Reservoir", Reservoir)
CobaRegistry.register("Shuffle"  , Shuffle  )
CobaRegistry.register("Sort"     , Sort     )
CobaRegistry.register("Scale"    , Scale    )
CobaRegistry.register("Cycle"    , Cycle    )
CobaRegistry.register("Where"    , Where    )
CobaRegistry.register("Repr"     , Repr     )

CobaRegistry.register("L1", L1Reward      )
CobaRegistry.register("HR", HammingReward )
CobaRegistry.register("BR", BinaryReward  )
CobaRegistry.register("DR", DiscreteReward)
