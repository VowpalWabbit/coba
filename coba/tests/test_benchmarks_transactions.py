import unittest

from statistics import mean

from coba.benchmarks.results import Result
from coba.benchmarks.transactions import Transaction, TransactionIsNew

class TransactionIsNew_Test(unittest.TestCase):
    
    def test_duplicates_are_dropped(self):
        existing = Result.from_transactions([
            Transaction.learner(0, a='A'),
            Transaction.simulation(0, b='B'),
            Transaction.batch(0, 1, reward=mean([1,2,3]))
        ])

        filter = TransactionIsNew(existing)

        transactions = list(filter.filter([
            Transaction.learner(0, a='A'), 
            Transaction.simulation(0, b='B'), 
            Transaction.batch(0, 1, reward=mean([1,2,3]))]
        ))

        self.assertEqual(len(transactions), 0)

    def test_non_duplicates_are_kept(self):
        existing = Result.from_transactions([
            Transaction.learner(0, a='A'),
            Transaction.simulation(0, b='B'),
            Transaction.batch(0, 1, reward=mean([1,2,3]))
        ])

        filter = TransactionIsNew(existing)

        transactions = list(filter.filter([
            Transaction.learner(1, a='A'), 
            Transaction.simulation(1, b='B'), 
            Transaction.batch(1, 1, reward=mean([1,2,3]))]
        ))

        self.assertEqual(len(transactions), 3)

if __name__ == '__main__':
    unittest.main()