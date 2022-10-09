import unittest

from collections import Counter

from coba.exceptions import CobaException
from coba.contexts import CobaContext, NullLogger

from coba.environments import ToInteractionGrounded, SimulatedInteraction

CobaContext.logger = NullLogger()

class ToInteractionGrounded_Tests(unittest.TestCase):
    
    def test_bad_users(self):
        with self.assertRaises(CobaException) as e:
            ToInteractionGrounded(10,20,5,2,1)

        self.assertIn("Igl conversion can't have more normal users", str(e.exception))

    def test_bad_words(self):
        with self.assertRaises(CobaException) as e:
            ToInteractionGrounded(10,5,5,10,1)

        self.assertIn("Igl conversion can't have more good words", str(e.exception))

    def test_fixed_with_respect_to_seed(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[1,0,0],c=0),
            SimulatedInteraction(1,[1,2,3],[0,1,0],c=1),
        ]
        to_igl_filter    = ToInteractionGrounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        self.assertEqual(True,igl_interactions[0].kwargs['isnormal'])
        self.assertEqual(4, igl_interactions[0].kwargs['userid'])
        self.assertEqual([(0,),(3,),(2,)], igl_interactions[0].kwargs['feedbacks'])

        self.assertEqual(False,igl_interactions[1].kwargs['isnormal'])
        self.assertEqual(9, igl_interactions[1].kwargs['userid'])
        self.assertEqual([(1,),(2,),(1,)], igl_interactions[1].kwargs['feedbacks'])

    def test_number_context_01_rewards(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[1,0,0],c=0),
            SimulatedInteraction(1,[1,2,3],[0,1,0],c=1),
            SimulatedInteraction(2,[1,2,3],[0,0,1],c=2),
        ]
        to_igl_filter    = ToInteractionGrounded(10,5,4,2,1)
        igl_interactions = list(to_igl_filter.filter(interactions*1000))

        self.assertEqual(3000, len(igl_interactions))

        normal_count = 0
        bizaro_count = 0

        word_counts = Counter()

        for interaction in igl_interactions:
            normal_count += int(interaction.kwargs['isnormal'])
            bizaro_count += int(not interaction.kwargs['isnormal'])
            word_counts  += Counter(interaction.kwargs['feedbacks'])

            self.assertEqual(interaction.context, (interaction.kwargs['userid'], interaction.kwargs['c']))
            self.assertEqual(interaction.kwargs['isnormal'],interaction.kwargs['userid'] in to_igl_filter.normalids)
            
            for word,reward in zip(interaction.rewards, interaction.kwargs['feedbacks']):
                if reward == 1: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.good_words)
                if reward == 0: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.bad_words)

        self.assertAlmostEqual(1,normal_count/bizaro_count,2)

        for word,count in word_counts.items():
            self.assertAlmostEqual(1/4,count/sum(word_counts.values()),2)

    def test_list_context_01_rewards(self):
        interactions = [
            SimulatedInteraction([0],[1,2,3],[1,0,0],c=[0]),
            SimulatedInteraction([1],[1,2,3],[0,1,0],c=[1]),
            SimulatedInteraction([2],[1,2,3],[0,0,1],c=[2]),
        ]
        to_igl_filter    = ToInteractionGrounded(10,5,4,2,1)
        igl_interactions = list(to_igl_filter.filter(interactions*1000))

        self.assertEqual(3000, len(igl_interactions))

        normal_count = 0
        bizaro_count = 0

        word_counts = Counter()

        for interaction in igl_interactions:
            normal_count += int(interaction.kwargs['isnormal'])
            bizaro_count += int(not interaction.kwargs['isnormal'])
            word_counts  += Counter(interaction.kwargs['feedbacks'])

            self.assertEqual(interaction.context, tuple([interaction.kwargs['userid']]+interaction.kwargs['c']))
            self.assertEqual(interaction.kwargs['isnormal'],interaction.kwargs['userid'] in to_igl_filter.normalids)
            
            for word,reward in zip(interaction.rewards, interaction.kwargs['feedbacks']):
                if reward == 1: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.good_words)
                if reward == 0: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.bad_words)

        self.assertAlmostEqual(1,normal_count/bizaro_count,2)

        for word,count in word_counts.items():
            self.assertAlmostEqual(1/4,count/sum(word_counts.values()),2)

    def test_dict_context_01_rewards(self):
        interactions = [
            SimulatedInteraction({'a':0},[1,2,3],[1,0,0],c={'a':0}),
            SimulatedInteraction({'b':1},[1,2,3],[0,1,0],c={'b':1}),
            SimulatedInteraction({'c':2},[1,2,3],[0,0,1],c={'c':2}),
        ]
        to_igl_filter    = ToInteractionGrounded(10,5,4,2,1)
        igl_interactions = list(to_igl_filter.filter(interactions*1000))

        self.assertEqual(3000, len(igl_interactions))

        normal_count = 0
        bizaro_count = 0

        word_counts = Counter()

        for interaction in igl_interactions:
            normal_count += int(interaction.kwargs['isnormal'])
            bizaro_count += int(not interaction.kwargs['isnormal'])
            word_counts  += Counter(interaction.kwargs['feedbacks'])

            self.assertEqual(interaction.context, dict(userid=interaction.kwargs['userid'],**interaction.kwargs['c']))
            self.assertEqual(interaction.kwargs['isnormal'],interaction.kwargs['userid'] in to_igl_filter.normalids)
            
            for word,reward in zip(interaction.rewards, interaction.kwargs['feedbacks']):
                if reward == 1: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.good_words)
                if reward == 0: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.bad_words)

        self.assertAlmostEqual(1,normal_count/bizaro_count,2)

        for word,count in word_counts.items():
            self.assertAlmostEqual(1/4,count/sum(word_counts.values()),2)

    def test_number_context_not_01_rewards(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[.1,0,0],c=0),
            SimulatedInteraction(1,[1,2,3],[0,.1,0],c=1),
            SimulatedInteraction(2,[1,2,3],[0,0,.1],c=2),
        ]
        to_igl_filter    = ToInteractionGrounded(10,5,4,2,1)
        igl_interactions = list(to_igl_filter.filter(interactions*1000))

        self.assertEqual(3000, len(igl_interactions))

        normal_count = 0
        bizaro_count = 0

        word_counts = Counter()

        for interaction in igl_interactions:
            normal_count += int(interaction.kwargs['isnormal'])
            bizaro_count += int(not interaction.kwargs['isnormal'])
            word_counts  += Counter(interaction.kwargs['feedbacks'])

            self.assertEqual(interaction.context, (interaction.kwargs['userid'], interaction.kwargs['c']))
            self.assertEqual(interaction.kwargs['isnormal'],interaction.kwargs['userid'] in to_igl_filter.normalids)
            
            for word,reward in zip(interaction.rewards, interaction.kwargs['feedbacks']):
                if reward == 1: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.good_words)
                if reward == 0: self.assertEqual(interaction.kwargs['isnormal'], word in to_igl_filter.bad_words)

        self.assertAlmostEqual(1,normal_count/bizaro_count,2)

        for word,count in word_counts.items():
            self.assertAlmostEqual(1/4,count/sum(word_counts.values()),2)

    def test_params(self):

        params = ToInteractionGrounded(10,5,4,2,1).params

        self.assertEqual(10, params['n_users'])
        self.assertEqual( 5, params['n_normal'])
        self.assertEqual( 4, params['n_words'])
        self.assertEqual( 2, params['n_good'])
        self.assertEqual( 1, params['igl_seed'])

if __name__ == '__main__':
    unittest.main()
