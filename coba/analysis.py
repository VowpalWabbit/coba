"""The analysis module contains functionality to help with analyzing Results."""

import math

from collections import defaultdict
from itertools import groupby
from typing import cast, Dict, List

from coba.utilities import check_matplotlib_support
from coba.benchmarks import Result
from coba.statistics import OnlineMean, OnlineVariance

class Plots():

    @staticmethod
    def standard_plot(result: Result, show_err: bool = False, show_sd: bool = False) -> None:

        def _plot(axes, label, xs, ys, vs, ns):
            axes.plot(xs, ys, label=label)

            if show_sd:
                ls = [ y-math.sqrt(v) for y,v in zip(ys,vs) ]
                us = [ y+math.sqrt(v) for y,v in zip(ys,vs) ]
                axes.fill_between(xs, ls, us, alpha = 0.1)

            if show_err:
                # I don't really understand what this is... For each x our distribution
                # is changing so its VAR is also changing. What does it mean to calculate
                # sample variance from a deterministic collection of random variables with
                # different distributions? For example sample variance of 10 random variables
                # from dist1 and 10 random variables from dist2... This is not the same as 20
                # random variables with 50% chance drawing from dist1 and 50% chance of drawing
                # from dist2. So the distribution can only be defined over the whole space (i.e.,
                # all 20 random variables) and not for a specific random variable. Oh well, for
                # now I'm leaving this as it is since I don't have any better ideas. I think what
                # I've done is ok, but I need to more some more thought into it.
                ls = [ y-math.sqrt(v/n) for y,v,n in zip(ys,vs,ns) ]
                us = [ y+math.sqrt(v/n) for y,v,n in zip(ys,vs,ns) ]
                axes.fill_between(xs, ls, us, alpha = 0.1)

        learners, _, batches = result.to_indexed_tuples()

        learner_index_key = lambda batch: (batch.learner_id, batch.batch_index)
        sorted_batches    = sorted(batches.values(), key=learner_index_key)
        grouped_batches   = groupby(groupby(sorted_batches , key=learner_index_key), key=lambda x: x[0][0])

        max_batch_N = 0

        indexes     = cast(Dict[int,List[int  ]], defaultdict(list))
        incounts    = cast(Dict[int,List[int  ]], defaultdict(list))
        inmeans     = cast(Dict[int,List[float]], defaultdict(list))
        invariances = cast(Dict[int,List[float]], defaultdict(list))
        cucounts    = cast(Dict[int,List[int  ]], defaultdict(list))
        cumeans     = cast(Dict[int,List[float]], defaultdict(list))
        cuvariances = cast(Dict[int,List[float]], defaultdict(list))

        for learner_id, learner_batches in grouped_batches:

            cucount    = 0
            cumean     = OnlineMean()
            cuvariance = OnlineVariance()

            for (_, batch_index), index_batches in learner_batches:

                incount    = 0
                inmean     = OnlineMean()
                invariance = OnlineVariance()

                for N, reward in [ (b.N, b.reward) for b in index_batches]:
                    
                    max_batch_N = max(N, max_batch_N)
                    
                    incount     = incount + 1
                    inmean      .update(reward)
                    invariance  .update(reward)
                    cucount     = cucount + 1
                    cumean      .update(reward)
                    cuvariance  .update(reward)

                #sanity check, sorting above (in theory) should take care of this...
                #if this isn't the case then the cu* values will be incorrect...
                assert indexes[learner_id] == [] or batch_index > indexes[learner_id][-1]

                incounts[learner_id].append(incount)
                indexes[learner_id].append(batch_index)
                inmeans[learner_id].append(inmean.mean)
                invariances[learner_id].append(invariance.variance)
                cucounts[learner_id].append(cucount)
                cumeans[learner_id].append(cumean.mean)
                cuvariances[learner_id].append(cuvariance.variance)

        check_matplotlib_support('Plots.standard_plot')
        import matplotlib.pyplot as plt #type: ignore

        fig = plt.figure()

        index_unit = "Interaction" if max_batch_N ==1 else "Batch"
        
        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for learner_id in learners:
            _plot(ax1, learners[learner_id].full_name, indexes[learner_id], inmeans[learner_id], invariances[learner_id], incounts[learner_id])

        ax1.set_title(f"Instantaneous Reward")
        ax1.set_ylabel("Mean Reward")
        ax1.set_xlabel(f"{index_unit} Index")

        for learner_id in learners:
            _plot(ax2, learners[learner_id].full_name, indexes[learner_id], cumeans[learner_id], cuvariances[learner_id], cucounts[learner_id])

        ax2.set_title("Cumulative Reward")
        ax1.set_xlabel(f"{index_unit} Index")

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