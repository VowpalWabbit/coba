from coba.tools import CobaRegistry, DiskCache, NoneCache, MemoryCache, ConsoleLog, NoneLog
from coba.simulations import OpenmlSimulation, PcaSimulation, ShuffleSimulation, TakeSimulation
from coba.benchmarks import BenchmarkFileFmtV1

CobaRegistry.register("ConsoleLog" , ConsoleLog )
CobaRegistry.register("NoneLog"    , NoneLog    )
CobaRegistry.register("DiskCache"  , DiskCache  )
CobaRegistry.register("NoneCache"  , NoneCache  )
CobaRegistry.register("MemoryCache", MemoryCache)

CobaRegistry.register("OpenmlSimulation", OpenmlSimulation )
CobaRegistry.register("Take"            , TakeSimulation   )
CobaRegistry.register("Shuffle"         , ShuffleSimulation)
CobaRegistry.register("PCA"             , PcaSimulation    )

CobaRegistry.register("BenchmarkFileV1", BenchmarkFileFmtV1)