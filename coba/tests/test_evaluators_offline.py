import json
import math
import unittest
import unittest.mock
import warnings

from coba.environments import Noise
from coba.environments import SupervisedSimulation
from coba.pipes import Pipes
from coba.utilities import PackageChecker

from coba.evaluators import ClassMetaEvaluator

class MulticlassMetaEvaluator_Tests(unittest.TestCase):

    def test_classification_statistics_empty_interactions(self):
        simulation = SupervisedSimulation([],[])
        row        = ClassMetaEvaluator().evaluate(simulation,None)
        self.assertEqual(row, {})

    def test_classification_statistics_dense_sans_sklearn(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            simulation = SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10)
            row        = ClassMetaEvaluator().evaluate(simulation,None)

            self.assertEqual(2, row["class_count"])
            self.assertEqual(2, row["feature_count"])
            self.assertEqual(0, row["class_imbalance_ratio"])

    def test_classification_statistics_sparse_sans_sklearn(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            c1 = [{"1":1, "2":2}, "A"]
            c2 = [{"1":3, "2":4}, "B"]

            simulation = SupervisedSimulation(*zip(*[c1,c2]*10))
            row        = ClassMetaEvaluator().evaluate(simulation,None)

            self.assertEqual(2, row["class_count"])
            self.assertEqual(2, row["feature_count"])
            self.assertEqual(0, row["class_imbalance_ratio"])

    def test_classification_statistics_encodable_sans_sklearn(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            c1 = [{"1":1,"2":2}, "A" ]
            c2 = [{"1":3,"2":4}, "B" ]

            simulation = SupervisedSimulation(*zip(*[c1,c2]*10))
            row        = ClassMetaEvaluator().evaluate(simulation,None)

            json.dumps(row)

    @unittest.skipUnless(PackageChecker.sklearn(strict=False), "sklearn is not installed so we must skip this test.")
    @unittest.skipUnless(PackageChecker.scipy(strict=False), "this test requires scipy")
    def test_classification_statistics_encodable_with_sklearn(self):
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

        simulation = Pipes.join(SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10),Noise())
        row        = ClassMetaEvaluator().evaluate(simulation,None)

        json.dumps(row)

    @unittest.skipUnless(PackageChecker.sklearn(strict=False), "sklearn is not installed so we must skip this test.")
    def test_classification_statistics_dense(self):
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

        simulation = Pipes.join(SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10),Noise())
        row        = ClassMetaEvaluator().evaluate(simulation,None)

        self.assertEqual(2, row["class_count"])
        self.assertEqual(2, row["feature_count"])
        self.assertEqual(0, row["class_imbalance_ratio"])

    @unittest.skipUnless(PackageChecker.sklearn(strict=False), "sklearn is not installed so we must skip the sklearn test")
    def test_classification_statistics_sparse(self):
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

        simulation = Pipes.join(SupervisedSimulation([{"1":1,"2":2},{"3":3,"4":4}]*10,["A","B"]*10), Noise())
        row        = ClassMetaEvaluator().evaluate(simulation,None)

        self.assertEqual(2, row["class_count"])
        self.assertEqual(4, row["feature_count"])
        self.assertEqual(0, row["class_imbalance_ratio"])

    def test_entropy(self):
        a = [1,2,3,4,5,6]
        b = [1,1,1,1,1,1]
        c = [1,2,1,2,1,2]
        self.assertAlmostEqual(math.log2(len(a)), ClassMetaEvaluator()._entropy(a))
        self.assertAlmostEqual(                0, ClassMetaEvaluator()._entropy(b))
        self.assertAlmostEqual(                1, ClassMetaEvaluator()._entropy(c))

    def test_entropy_normed(self):
        a = [1,2,3,4,5]
        b = [1,1,1,1,1]
        self.assertAlmostEqual(1, ClassMetaEvaluator()._entropy_normed(a))
        self.assertAlmostEqual(0, ClassMetaEvaluator()._entropy_normed(b))

    def test_mutual_info(self):
        #mutual info, I(), tells me how many bits of info two random variables convey about eachother
        #entropy, H(), tells me how many bits of info are needed in order to fully know a random variable
        #therefore, if knowing a random variable x tells me everything about y then I(x;y) == H(y)
        #this doesn't mean that y necessarily tells me everything about x (i.e., I(x;y) may not equal H(x))

        #In classification then I don't really care about how much y tells me about x... I do care how much
        #X tells me about y so I(x;y)/H(y) tells me what percentage of necessary bits x gives me about y.
        #This explains explains the equivalent num of attributes feature since \sum I(x;y) must be >= H(y)
        #for the dataset to be solvable.

        #I also care about how much information multiple features of x give me about eachother. That's
        #because if all x tell me about eachother than what might look like a lot of features actually
        #contains very little information. So, in that case I think I want 2*I(x1;x2)/(H(x1)+H(x2)). If
        #H(x1) >> H(x2) and I(x1;x2) == H(x2) then my above formulation will hide the fact that x1 tells
        #me everything about x2 but I think that's ok because it still tells me that taken as a pair there's
        #still a lot of information.

        #how much information does each X give me about y (1-H(y|X)/H(y)==I(y;x)/H(y) with 0 meaning no info)
        #how much information does x2 give about x1? 1-H(x1|x2)/H(x1)

        #x1 tells me z1 bits about y , x2 tells me z2 bits about y
        #x1 tells me w1 bits about x2, x2 tells me w2 bits about x1

        #in this case knowing x or y tells me nothing about the other so mutual info is 0
        #P(b|a) = P(b)
        #P(a|b) = P(a)
        #H(b) = 0; H(b|a) = 0; H(a,b) = 2; I(b;a) = 0
        #H(a) = 2; H(a|b) = 2; H(a,b) = 2; I(a;b) = 0
        a = [1,2,3,4]
        b = [1,1,1,1]
        self.assertAlmostEqual(0, ClassMetaEvaluator()._mutual_info(a,b))

        #H(b) = 1; H(b|a) = 0; H(a,b) = 1; I(b;a) = 1
        #H(a) = 1; H(a|b) = 0; H(a,b) = 1; I(a;b) = 1
        a = [1,2,1,2]
        b = [1,1,2,2]
        self.assertAlmostEqual(0, ClassMetaEvaluator()._mutual_info(a,b))

        a = [1,2,1,2]
        b = [2,1,2,1]
        self.assertAlmostEqual(1, ClassMetaEvaluator()._mutual_info(a,b))

        a = [1,2,3,4]
        b = [1,2,3,4]
        self.assertAlmostEqual(2, ClassMetaEvaluator()._mutual_info(a,b))

    def test_dense(self):

        X = [[1,2,3],[4,5,6]]
        self.assertEqual(X, ClassMetaEvaluator()._dense(X))

        X = [{'a':1}, {'b':2}, {'a':3, 'b':4}]
        self.assertEqual([[1,0],[0,2],[3,4]], ClassMetaEvaluator()._dense(X))

    def test_bin(self):

        X = [[1,2],[2,3],[3,4],[4,5]]
        self.assertEqual([[0,0],[0,0],[1,1],[1,1]], ClassMetaEvaluator()._bin(X,2))

        X = [[1,2],[2,3],[3,4],[4,5]]
        self.assertEqual([[0,0],[1,1],[1,1],[2,2]], ClassMetaEvaluator()._bin(X,3))

        X = [[1,2],[2,3],[3,4],[4,5]]
        self.assertEqual([[0,0],[1,1],[2,2],[3,3]], ClassMetaEvaluator()._bin(X,4))

    def test_imbalance_ratio_1(self):

        self.assertAlmostEqual(0, ClassMetaEvaluator()._imbalance_ratio([1,1,2,2]))
        self.assertAlmostEqual(1, ClassMetaEvaluator()._imbalance_ratio([1,1]))
        self.assertIsNone     (   ClassMetaEvaluator()._imbalance_ratio([]))

    def test_volume_overlapping_region(self):

        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.04, ClassMetaEvaluator()._volume_overlapping_region(X,Y))

    def test_max_individual_feature_efficiency(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.5, ClassMetaEvaluator()._max_individual_feature_efficiency(X,Y))

    @unittest.skipUnless(PackageChecker.numpy(strict=False), "numpy is not installed so we must skip this test.")
    def test_max_individual_feature_efficiency(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.5, ClassMetaEvaluator()._max_individual_feature_efficiency(X,Y))

    @unittest.skipUnless(not PackageChecker.numpy(strict=False), "numpy is installed so we must skip this test.")
    def test_max_individual_feature_efficiency_sans_numpy(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertIsNone(ClassMetaEvaluator()._max_individual_feature_efficiency(X,Y))

    @unittest.skipUnless(PackageChecker.sklearn(strict=False), "sklearn is not installed so we must skip this test.")
    def test_max_directional_fisher_discriminant_ratio(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.529, ClassMetaEvaluator()._max_directional_fisher_discriminant_ratio(X,Y), places=3)

if __name__ == '__main__':
    unittest.main()
