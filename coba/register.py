from coba.tools import CobaRegistry, DiskCache, NoneCache, MemoryCache, ConsoleLog, NoneLog
from coba.simulations import OpenmlSimulation, PCA, Shuffle, Take
from coba.benchmarks import BenchmarkFileFmtV1, BenchmarkFileFmtV2

CobaRegistry.register("ConsoleLog" , ConsoleLog )
CobaRegistry.register("NoneLog"    , NoneLog    )
CobaRegistry.register("DiskCache"  , DiskCache  )
CobaRegistry.register("NoneCache"  , NoneCache  )
CobaRegistry.register("MemoryCache", MemoryCache)

CobaRegistry.register("OpenmlSimulation", OpenmlSimulation)
CobaRegistry.register("Take"            , Take            )
CobaRegistry.register("Shuffle"         , Shuffle         )
CobaRegistry.register("PCA"             , PCA             )

CobaRegistry.register("BenchmarkFileV1", BenchmarkFileFmtV1)
CobaRegistry.register("BenchmarkFileV2", BenchmarkFileFmtV2)