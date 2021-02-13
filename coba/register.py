from coba.tools import register_class, DiskCache, NoneCache, MemoryCache, ConsoleLog, NoneLog
from coba.simulations import OpenmlSimulation, PcaSimulation, ShuffleSimulation, TakeSimulation
from coba.benchmarks import BenchmarkFileFmtV1

register_class("ConsoleLog" , ConsoleLog )
register_class("NoneLog"    , NoneLog    )
register_class("DiskCache"  , DiskCache  )
register_class("NoneCache"  , NoneCache  )
register_class("MemoryCache", MemoryCache)

register_class("OpenmlSimulation", OpenmlSimulation )
register_class("Take"            , TakeSimulation   )
register_class("Shuffle"         , ShuffleSimulation)
register_class("PCA"             , PcaSimulation    )

register_class("BenchmarkFileV1", BenchmarkFileFmtV1)