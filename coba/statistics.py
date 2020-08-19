"""The statistics module contains algorithms and methods to calculate statistics.

TODO Add unittests to make sure weighted_mean works correctly
"""

import math

from typing import Sequence, Iterable
from statistics import mean, variance

class OnlineVariance():
    """Calculate sample variance in an online fashion.
    
    Remarks:
        This algorithm is known as Welford's algorithm and the implementation below
        is a modified version of the Python algorithm by Wikepedia contirubtors (2020).

    References:
        Wikipedia contributors. (2020, July 6). Algorithms for calculating variance. In Wikipedia, The
        Free Encyclopedia. Retrieved 18:00, July 24, 2020, from 
        https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=966329915
    """

    def __init__(self) -> None:
        """Instatiate an OnlineVariance calcualator."""
        self._count    = 0.
        self._mean     = 0.
        self._M2       = 0.
        self._variance = float("nan")

    @property
    def variance(self) -> float:
        """The variance of all given updates."""
        return self._variance

    def update(self, value: float) -> None:
        """Update the current variance with the given value."""
        
        (count,mean,M2) = (self._count, self._mean, self._M2)

        count   += 1
        delta   = value - mean
        mean   += delta / count
        delta2  = value - mean
        M2     += delta * delta2

        (self._count, self._mean, self._M2) = (count, mean, M2)

        if count > 1:
            self._variance = M2 / (count - 1)

class OnlineMean():
    """Calculate mean in an online fashion."""

    def __init__(self):
        self._n = 0
        self._mean = float('nan')

    @property
    def mean(self) -> float:
        """The mean of all given updates."""

        return self._mean

    def update(self, value:float) -> None:
        """Update the current mean with the given value."""
        
        self._n += 1

        alpha = 1/self._n

        self._mean = value if alpha == 1 else (1 - alpha) * self._mean + alpha * value

class StatisticalEstimate:

    def __init__(self, N:int, estimate:float, standard_error: float) -> None:
        self._N              = N
        self._estimate       = estimate
        self._standard_error = standard_error

    @property
    def N(self) -> int:
        return self._N

    @property
    def estimate(self) -> float:
        return self._estimate
    
    @property
    def standard_error(self) -> float:
        return self._standard_error

class BatchMeanEstimator(StatisticalEstimate):

    def __init__(self, sample: Sequence[float]) -> None:
        N              = len(sample)
        estimate       = mean(sample) if len(sample) > 0 else float('nan')
        standard_error = math.sqrt(variance(sample)/len(sample)) if len(sample) > 1 else float('nan')

        super().__init__(N, estimate, standard_error)

class Aggregators:
    """Methods of aggregating statistics across simulations.
    
    Remarks:
        In theory, if every StatisticalEstimate kept their entire sample then we could estimate values
        empirically by mixing all samples into a single big pot and then recalculating mean, and SEM.
        Unfortunately, this approach while simple has two drawbacks. First, if a benchmark has many simulations
        with many interactions this means that our package will always be memory constrained to O(n) where n is
        the number of interactions in the benchmark. Second, our computation complexity would also be O(n) to
        since blending simulation statistics would require relooping over all samples again. There are downsides to
        not taking the simple approach but the reduction in memory and computational complexity seem to be worth it. 

        One downside of not taking the simple approach is that stats with N == 1 can become problematic since we don't 
        have an estimate of their standard error. Therefore we adjust for this below by dividing the standard error by 
        (N - int(N==1)). This adjustment means that standard error can become biased but hopefully not by too much.
    """
    
    @staticmethod
    def unweighted_mean(stats: Iterable[StatisticalEstimate]) -> StatisticalEstimate:
        """Calculate the unweighted mean and standard error from a collection of statistical estimates.

        Args:
            stats: Previously calculated statistical estimates that we wish to average.
        
        WARNING:
            This was not done in an incredibly rigorous fashion. There are likely better means of estimating.
        """

        total_sum = 0.
        total_var = 0.
        total_N   = 0
        total_no_se = 0 

        for stat in stats:
            total_N     += stat.N
            total_sum   += stat.estimate * stat.N if not math.isnan(stat.estimate) else 0
            total_var   += (stat.standard_error**2)*(stat.N**2) if not math.isnan(stat.standard_error) else 0
            total_no_se += int(math.isnan(stat.standard_error))

        N              = total_N
        mean           = total_sum/total_N
        standard_error = math.sqrt(total_var)/(total_N-total_no_se) if total_N-total_no_se > 0 else float('nan')

        return StatisticalEstimate(N, mean, standard_error)

    @staticmethod
    def weighted_mean(stats: Iterable[StatisticalEstimate]) -> StatisticalEstimate:
            """Calculate the mean statistic and stnadard error from a collection of statistical estimates.

            Args:
                stats: Previously calculated statistical estimates that we wish to average.
            
            WARNING:
                This was not done in an incredibly rigorous fashion. There are likely better means of estimating.
            """

            total_sum = 0.
            total_var = 0.
            total_N   = 0
            total_no_se = 0 

            for stat in stats:
                total_N     += stat.N
                total_sum   += stat.estimate * stat.N if not math.isnan(stat.estimate) else 0
                total_var   += (stat.standard_error**2)*(stat.N**2) if not math.isnan(stat.standard_error) else 0
                total_no_se += int(math.isnan(stat.standard_error))

            N              = total_N
            mean           = total_sum/total_N
            standard_error = math.sqrt(total_var)/(total_N-total_no_se) if total_N-total_no_se > 0 else float('nan')

            return StatisticalEstimate(N, mean, standard_error)