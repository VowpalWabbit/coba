"""The statistics module contains algorithms and methods to calculate statistics.

TODO Add unittests to make sure SummaryStats.blend works correctly
TODO Add unittests to make sure SummaryStats.blend_cumulative works correctly
"""

import math

from typing import Sequence, Iterable, List

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

class SummaryStats:
    """A class to store summary statistics calculated from some sample."""

    @staticmethod
    def from_observations(observations: Sequence[float]) -> 'SummaryStats':
        return SummaryStats().add_observations(observations)

    @staticmethod
    def blend(stats: Iterable['SummaryStats']) -> 'SummaryStats':
        """Calculate the resulting stats from blending the SummaryStats of several samples.

        Args:
            stats: The previously calculated stats that we wish to merge.

        Remarks:
            In theory, if every 'Stats' object kept their entire sample then we could estimate all the
            values that we calculate below empirically by mixing all samples into a single big pot and
            then recalculating mean, variance and SEM. Unfortunately, this approach while simple has two
            drawbacks. First, if a benchmark has many simulations with many interactions this means that 
            our package will always be constrained since our memory would be O(n) to interactions. Second, our 
            computation complexity would also be O(n) to interactions since merging stats would require relooping 
            over all samples again. One downside of not taking the simple approach is that stats with N == 1 
            become problematic since we don't know their variance. Therefore we adjust for this below by 
            adding the average variance back in for ever stat with N == 1.
        """

        total_sum = 0.
        total_var = 0.
        total_N   = 0
        total_N_1 = 0 

        for stat in stats: #type: ignore #(this error is a bug with pylance)
            total_sum += stat.mean * stat.N if stat.N > 0 else 0
            total_var += (stat.SEM**2)*(stat.N**2) if stat.N > 1 else 0
            total_N   += stat.N
            total_N_1 += int(stat.N == 1)

        #when we only have a single observation there is no way for us to estimate
        #the variance of that random number. Therefore we add a neutral amount of
        #variance since this number is still in total and total_N and we don't want
        #to remove it.
        total_var += (total_var/total_N) * total_N_1

        N        = total_N
        mean     = total_sum/total_N
        variance = float('nan') # once we start to blend samples there's no longer a meaningful way to calculate variance
        SEM      = math.sqrt(total_var/total_N**2)

        return SummaryStats(N, mean, variance, SEM)

    @staticmethod
    def blend_cumulative(stats: Iterable['SummaryStats']) -> Sequence['SummaryStats']:
        """Calculate the resulting stats from cumulatively blending the SummaryStats of several samples.

        Args:
            stats: The previously calculated stats that we wish to blend cumulatively.

        Remarks:
            For more information see `blend`.
        """

        total_sum = 0.
        total_var = 0.
        total_N   = 0

        cumulative_stats: List[SummaryStats] = []

        for stat in stats: #type: ignore #(this error is a bug with pylance)
            total_N   += stat.N
            total_sum += stat.mean * stat.N if stat.N > 0 else 0
            total_var += (stat.SEM**2)*(stat.N**2) if stat.N > 1 else total_var/total_N

            N    = total_N
            mean = total_sum/total_N
            SEM  = math.sqrt(total_var/total_N**2)

            cumulative_stats.append(SummaryStats(N=N, mean=mean, SEM=SEM))

        return cumulative_stats



    def __init__(self, N: int = 0, mean: float = float('nan'), variance:float = float('nan'), SEM: float = float('nan')):
        """Instantiate a Stats class.

        Args:
            N: The size of the sample of interest.
            mean: The mean for some sample of interest.
            variance: The variance for some sample of interest.
            SEM: The standard error of the mean for some sample of interest.
        """

        self._online_mean     = OnlineMean()
        self._online_variance = OnlineVariance()

        self._N    = N
        self._mean = mean
        self._var  = variance
        self._SEM  = SEM

    @property
    def N(self) -> int:
        """The size of the sample."""
        return self._N

    @property
    def mean(self) -> float:
        """The mean for some sample."""
        return self._mean

    @property
    def variance(self) -> float:
        """The mean for some sample."""
        return self._var

    @property
    def SEM(self) -> float:
        """The mean for some sample."""
        return self._SEM

    def add_observations(self, observations: Sequence[float]) -> 'SummaryStats':

        for observation in observations:
            self._online_mean.update(observation)
            self._online_variance.update(observation)

        self._N    = len(observations)
        self._mean = self._online_mean.mean
        self._var  = self._online_variance.variance
        self._SEM  = math.sqrt(self._var/self._N) if self._N > 0 else float('nan')

        return self