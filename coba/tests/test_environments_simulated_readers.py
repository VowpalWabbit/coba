import unittest

from pathlib import Path

from coba.pipes import MemoryIO, DiskIO, NullIO
from coba.contexts import CobaContext, NullLogger

from coba.environments import CsvSimulation, ArffSimulation, LibsvmSimulation, ManikSimulation

CobaContext.logger = NullLogger()

class CsvSimulation_Tests(unittest.TestCase):

    def test_memory_source(self):
        source       = MemoryIO(['a,b,c','1,2,3','4,5,6'])
        simulation   = CsvSimulation(source,'c')
        interactions = list(simulation.read())

        self.assertEqual(2, len(interactions))
        
        self.assertEqual(('1','2'), interactions[0].context)
        self.assertEqual(('4','5'), interactions[1].context)

        self.assertEqual(['3','6'], interactions[0].actions)
        self.assertEqual(['3','6'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])
    
    def test_file_path(self):

        Path("coba/tests/.temp/sim.csv").write_text("""
            a,b,c
            1,2,3
            4,5,6
        """)

        try:
            simulation   = CsvSimulation("coba/tests/.temp/sim.csv",'c')
            interactions = list(simulation.read())

            self.assertEqual(2, len(interactions))
            
            self.assertEqual(('1','2'), interactions[0].context)
            self.assertEqual(('4','5'), interactions[1].context)

            self.assertEqual(['3','6'], interactions[0].actions)
            self.assertEqual(['3','6'], interactions[1].actions)

            self.assertEqual([1,0], interactions[0].kwargs["rewards"])
            self.assertEqual([0,1], interactions[1].kwargs["rewards"])

        finally:
            Path("coba/tests/.temp/sim.csv").unlink()

    def test_params(self):
        self.assertEqual({'csv':'coba/tests/.temp/sim.csv'}, CsvSimulation('coba/tests/.temp/sim.csv', 'c').params)

class ArffSimulation_Tests(unittest.TestCase):

    def test_simple(self):

        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        source       = MemoryIO(lines)
        simulation   = ArffSimulation(source,'c')
        interactions = list(simulation.read())

        self.assertEqual(2, len(interactions))
        
        self.assertEqual((1,2), interactions[0].context)
        self.assertEqual((2,3), interactions[1].context)

        self.assertEqual(['0','class_B'], interactions[0].actions)
        self.assertEqual(['0','class_B'], interactions[1].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0], interactions[1].kwargs["rewards"])

    def test_one_hot(self):

        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B {0, 1, 2, 3}",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
            "3,1,0"
        ]

        source       = MemoryIO(lines)
        simulation   = ArffSimulation(source,'c',)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))
        
        self.assertEqual((1,(0,0,1,0)), interactions[0].context)
        self.assertEqual((2,(0,0,0,1)), interactions[1].context)
        self.assertEqual((3,(0,1,0,0)), interactions[2].context)

        self.assertEqual(['0','class_B'], interactions[0].actions)
        self.assertEqual(['0','class_B'], interactions[1].actions)
        self.assertEqual(['0','class_B'], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'arff':'coba/tests/.temp/sim.csv'}, ArffSimulation('coba/tests/.temp/sim.csv', 'c').params)

class LibsvmSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):

        lines = [
            "0 4:2 5:3",
            "1 1:1 2:1",
            "1 3:4"
        ]

        source       = MemoryIO(lines)
        simulation   = LibsvmSimulation(source)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))

        self.assertEqual({4:2,5:3}, interactions[0].context)
        self.assertEqual({1:1,2:1}, interactions[1].context)
        self.assertEqual({3:4    }, interactions[2].context)

        self.assertEqual(['0','1'], interactions[0].actions)
        self.assertEqual(['0','1'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'libsvm':'coba/tests/.temp/sim.csv'}, LibsvmSimulation('coba/tests/.temp/sim.csv').params)

class ManikSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):

        lines = [
            "Total_Points Num_Features Num_Labels",
            "0 4:2 5:3",
            "1 1:1 2:1",
            "1 3:4"
        ]

        source       = MemoryIO(lines)
        simulation   = ManikSimulation(source)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))

        self.assertEqual({4:2,5:3}, interactions[0].context)
        self.assertEqual({1:1,2:1}, interactions[1].context)
        self.assertEqual({3:4    }, interactions[2].context)

        self.assertEqual(['0','1'], interactions[0].actions)
        self.assertEqual(['0','1'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'manik':'coba/tests/.temp/sim.csv'}, ManikSimulation('coba/tests/.temp/sim.csv').params)



if __name__ == '__main__':
    unittest.main()
