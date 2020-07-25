"""The analysis module contains functionality to help with analyzing Results."""

from typing import Sequence

from coba.utilities import check_matplotlib_support
from coba.benchmarks import Result

class Plots():

    @staticmethod
    def standard_plot(results: Sequence[Result]) -> None:

        check_matplotlib_support('Plots.standard_plot')
        import matplotlib.pyplot as plt

        fig = plt.figure()

        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for result in results:
            ax1.plot([ i.mean for i in result .batch_stats], label=result.learner_name)

        ax1.set_title("Reward by Batch Index")
        ax1.set_ylabel("Mean Reward")
        ax1.set_xlabel("Batch Index")

        for result in results:
            ax2.plot([ i.mean for i in result.cumulative_batch_stats], label=result.learner_name)

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


