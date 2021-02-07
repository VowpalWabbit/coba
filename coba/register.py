from coba.registry import register_class
from coba.simulations import OpenmlSimulation, PcaSimulation, ShuffleSimulation, TakeSimulation
from coba.benchmarks import BenchmarkFileFmtV1

register_class("OpenmlSimulation", OpenmlSimulation)
register_class("Take", TakeSimulation)
register_class("Shuffle", ShuffleSimulation)
register_class("PCA", PcaSimulation)
register_class("BenchmarkFileV1", BenchmarkFileFmtV1)
