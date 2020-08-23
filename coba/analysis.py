"""The analysis module contains functionality to help with analyzing Results."""

from collections import defaultdict
from itertools import groupby
from typing import List, Dict, Sequence, Tuple

from coba.statistics import StatisticalEstimate
from coba.utilities import check_matplotlib_support
from coba.benchmarks import Result

class Plots():

    @staticmethod
    def standard_plot(result: Result) -> None:

        def plot(axes, label, estimates):
            x = [ i+1                          for i in range(len(estimates)) ]
            y = [ e.estimate                   for e in estimates             ]
            l = [ e.estimate-e.standard_error  for e in estimates             ]
            u = [ e.estimate+e.standard_error  for e in estimates             ]

            axes.plot(x,y, label=label)
            axes.fill_between(x, l, u, alpha = 0.25)

        def mean(weights: Sequence[float], means:Sequence[StatisticalEstimate]) -> StatisticalEstimate:
            weighted_sum = sum([w*m for w,m in zip(weights,means)])
            
            if isinstance(weighted_sum, StatisticalEstimate):
                return weighted_sum/sum(weights)
            else:
                return StatisticalEstimate(float('nan'), float('nan'))

        learners, _, batches = result.to_indexed_tuples()

        sorted_batches  = sorted(batches.values(), key=lambda r: (r.learner_id, r.batch_id))
        grouped_batches = groupby(sorted_batches , key=lambda r: (r.learner_id, r.batch_id))

        estimates: Dict[str,Tuple[List[float],List[StatisticalEstimate]]] = defaultdict(lambda: ([],[]))

        for batch_group in grouped_batches:
            name    = learners[batch_group[0][0]].full_name
            group   = list(batch_group[1])
            weights = [perf.N           for perf in group]
            means   = [perf.mean_reward for perf in group]            
            estimates[name][0].append(sum(weights))
            estimates[name][1].append(mean(weights, means))

        check_matplotlib_support('Plots.standard_plot')
        import matplotlib.pyplot as plt #type: ignore

        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for name, (weights, means) in estimates.items():
            plot(ax1, name, means)

        ax1.set_title("Reward by Batch Index")
        ax1.set_ylabel("Mean Reward")
        ax1.set_xlabel("Batch Index")

        for name, (weights, means) in estimates.items(): 
            plot(ax2, name, [ mean(weights[0:i+1], means[0:i+1]) for i in range(len(means)) ])

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
        fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .175), fancybox=True, ncol=3) #type: ignore

        plt.show()