from math import hypot, erf, sqrt
from statistics import fmean
from sys import version_info
from operator import mul, sub
from bisect import bisect_left
from itertools import repeat, accumulate, compress, chain
from typing import Sequence, Tuple, Union, Optional

def iqr(values: Sequence[float]) -> float:

    if len(values) <= 1: return 0.

    values = sorted(values)

    p25,p75 = percentile(values, [0.25,0.75])

    return p75-p25

def percentile(values: Sequence[float], percentiles: Union[float,Sequence[float]], weights: Sequence[float] = None, sort: bool = True) -> Union[float, Tuple[float,...]]:

    if len(values) == 1:
        if isinstance(percentiles,(int,float)):
            return values[0]
        else:
            return list(values)*len(percentiles)

    def _percentile(values: Sequence[float], weights:Optional[Sequence[float]], percentile: float) -> float:

        assert 0 <= percentile and percentile <= 1, "Percentile must be between 0 and 1 inclusive."

        if percentile == 0:
            return values[0]

        if percentile == 1:
            return values[-1]

        if weights:
            R = bisect_left(weights,percentile)
            L = R-1
            LP = (weights[R]-percentile)/(weights[R]-weights[L])

            return LP*values[L] + (1-LP)*values[R]
        else:
            i = percentile*(len(values)-1)
            I = int(i)

            if i == I:
                return values[I]
            else:
                w = (i-I)
                return (1-w)*values[I] + w*values[I+1]

    if sort:
        if weights:
            values, weights = zip(*sorted(zip(values,weights)))
        else:
            values = sorted(values)

    if weights:
        if any(not w for w in weights):
            values = list(compress(values,weights))
        weight_sum = sum(weights[1:])
        weights    = [w/weight_sum for w in accumulate(chain([0],compress(weights[1:],weights)))]

    if isinstance(percentiles,(float,int)):
        return _percentile(values, weights, percentiles)
    else:
        return tuple([_percentile(values, weights, p) for p in percentiles ])

def phi(x: float) -> float:
    'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def mean(sample: Sequence[float]) -> float:
    return fmean(sample)

def var(sample: Sequence[float]) -> float:
    n = len(sample)

    if n == 1: return float('nan')

    if version_info >= (3,8): #pragma: no cover
        #In Python 3.8 hypot was extended to support any number of items
        #this provides a fast/safe way to calculate the sum of squares
        #and so we revert to the one-pass method and trust in hypot
        E_s2 = hypot(*sample)**2
        E_s = sum(sample)
        return (E_s2-E_s*E_s/n)/(n-1)
    else: #pragma: no cover
        #using the corrected two pass algo as recommended by
        #https://cpsc.yale.edu/sites/default/files/files/tr222.pdf
        #I've optimized this as much as I think is possible in python
        diffs = tuple(map(sub,sample,repeat(sum(sample)/n)))
        return sum(map(mul,diffs,diffs))/(n-1)

def stdev(sample: Sequence[float]) -> float:
    return var(sample)**(1/2)

class OnlineVariance:
    """Calculate sample variance in an online fashion.

    Remarks:
        This algorithm is known as Welford's algorithm and the implementation below
        is a modified version of the Python algorithm created by Wikepedia contributors (2020).

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

class OnlineMean:
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
