"""The analysis module contains functionality to help with analyzing Results."""

import itertools
from math import isnan
from collections import defaultdict
from itertools import groupby, accumulate
from statistics import mean
from typing import List, Dict, Sequence, Tuple, cast

from coba.statistics import StatisticalEstimate
from coba.utilities import check_matplotlib_support
from coba.benchmarks import Result

class Plots():

    @staticmethod
    def standard_plot(result: Result, show_err: bool = False, weighted: bool = False) -> None:

        def _mean(weights, rewards) -> StatisticalEstimate:
            if weighted:
                return mean([w*r for w,r in zip(weights,rewards)]) #type: ignore
            else:
                return mean(rewards)

        def _cummean(weights, rewards) -> Sequence[StatisticalEstimate]:

            dens = list(accumulate(weights))
            nums = list(accumulate([w*r for w,r in zip(weights,rewards)]))

            return [n/d for n,d in zip(nums,dens)]

        def _plot(axes, label, x, estimates):
            y = [ e.estimate                   for e in estimates ]
            l = [ e.estimate-e.standard_error  for e in estimates ]
            u = [ e.estimate+e.standard_error  for e in estimates ]

            axes.plot(x,y, label=label)
            
            if show_err:
                axes.fill_between(x, l, u, alpha = 0.25)

        learners, _, batches = result.to_indexed_tuples()

        #For backwards compatability. Can be removed in future updates.
        if ('mean_reward' in list(batches.values())[0]._fields):
            reward = lambda batch: batch.mean_reward
        else:
            reward = lambda batch: batch.reward

        filter_predicate = lambda batch: (not isnan(reward(batch).standard_error) or not show_err)
        group_key        = lambda batch: (batch.learner_id, batch.batch_index)

        filtered_batches = filter(filter_predicate, batches.values())
        sorted_batches   = sorted(filtered_batches, key=group_key)
        grouped_batches  = groupby(sorted_batches , key=group_key)

        observations: Dict[int, Tuple[List[int], List[float], List[StatisticalEstimate]]] = defaultdict(lambda: ([],[],[]))

        for (learner_id, batch_index), batch_group in grouped_batches:

            weights, rewards = tuple(zip(*[ (batch.N, reward(batch)) for batch in batch_group]))

            observations[learner_id][0].append(batch_index)
            observations[learner_id][1].append(sum(weights))
            observations[learner_id][2].append(_mean(weights,rewards))

        check_matplotlib_support('Plots.standard_plot')
        import matplotlib.pyplot as plt #type: ignore

        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for learner_id, (indexes, weights, rewards) in observations.items():
            _plot(ax1, learners[learner_id].full_name, indexes, rewards)

        ax1.set_title("Reward by Batch Index")
        ax1.set_ylabel("Mean Reward")
        ax1.set_xlabel("Batch Index")

        for learner_id, (indexes, weights, rewards) in observations.items():
            _plot(ax2, learners[learner_id].full_name, indexes, _cummean(weights,rewards))

        ax2.set_title("Progressive Validation Reward")
        ax2.set_xlabel("Batch Index")

        (bot1, top1) = ax1.get_ylim()
        (bot2, top2) = ax2.get_ylim()

        ax1.set_ylim(min(bot1,bot2), max(top1,top2))
        ax2.set_ylim(min(bot1,bot2), max(top1,top2))

        scale = 0.25
        box1 = ax1.get_position()
        box2 = ax2.get_position()
        ax1.set_position([box1.x0, box1.y0 + box1.height * scale, box1.width, box1.height * (1-scale)])
        ax2.set_position([box2.x0, box2.y0 + box2.height * scale, box2.width, box2.height * (1-scale)])

        # Put a legend below current axis
        fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .175), fancybox=True, ncol=2) #type: ignore

        plt.show()