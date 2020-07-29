"""The statistics module contains algorithms and methods to calculate statistics."""

import math

from typing import Sequence

class OnlineVariance():
    """Calculate variance in an online fashion.
    
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

class Stats:
    """A class to store summary statistics calculated from some sample."""

    @staticmethod
    def from_observations(observations: Sequence[float]) -> 'Stats':
        """Create a Stats class for some given sequence of values.

        Args:
            vals: A sample of values to calculate statistics for.
        """

        online_mean = OnlineMean()
        online_var  = OnlineVariance()

        for observation in observations:
            online_mean.update(observation)
            online_var .update(observation)

        N    = len(observations)
        mean = online_mean.mean
        var  = online_var.variance
        SEM  = math.sqrt(var/N) if N > 0 else float('nan')

        return Stats(N, mean, var, SEM)

    def __init__(self, N: int = 0, mean: float = float('nan'), variance:float = float('nan'), SEM: float = float('nan')):
        """Instantiate a Stats class.

        Args:
            N: The size of the sample of interest.
            mean: The mean for some sample of interest.
            variance: The variance for some sample of interest.
            SEM: The standard error of the mean for some sample of interest.
        """

        self._N        = N
        self._mean     = mean
        self._variance = variance
        self._SEM      = SEM

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
        return self._variance

    @property
    def SEM(self) -> float:
        """The mean for some sample."""
        return self._SEM

    def blend(self, stats: 'Stats') -> None: #type: ignore #(this error is a bug with pylance)
        """Calculate the stats that would come from blending two samples.
        
        Args:
            stats: The previously calculated stats that we wish to merge.

        Remarks:
            In theory, if every 'Stats' object kept their entire sample then we could estimate all the
            values that we calculate below empirically by mixing all samples into a single big pot and
            then recalculating mean, variance and SEM. Unfortunately, this approach while simple has two
            drawbacks. First, if a benchmark has many simulations with many examples this would mean that 
            our package would always be constrained since our memory complexity would be O(n). Second, our 
            computation complexity would also be around O(Sn) since merging `S` stats will require relooping 
            over all samples again. The downside of not taking the simple approach is that stats with N == 1 
            become problematic since we don't know their variance. Therefore we adjust for this below by 
            adding the average variance back in for ever stat with N == 1.
        """

        total     = 0.
        total_var = 0.
        total_N   = 0
        total_N_1 = 0 

        for stat in [stats, self]: #type: ignore #(this error is a bug with pylance)
            total     += stat.mean * stat.N if stat.N > 0 else 0
            total_var += stat.variance * stat.N if stat.N > 1 else 0
            total_N   += stat.N
            total_N_1 += int(stat.N == 1)

        #when we only have a single observation there is no way for us to estimate
        #the variance of that random number therefore add in a neutral amount of
        #variance since this number is still in total and total_N and we don't want
        #to remove it.
        total_var += (total_var/total_N) * total_N_1

        #to understand why we calculate variance as we do below consider the following
        # E[Z] = (3/5)E[X] + (2/5)E[Y]
        # 5*Var[Z] = 3*Var[X] + 2*Var[Y]

        self._N        = total_N
        self._mean     = total/total_N
        self._variance = total_var/total_N
        self._SEM      = math.sqrt(total_var)/total_N

    def copy(self) -> 'Stats':
        return Stats(self._N, self._mean, self._variance, self._SEM)

