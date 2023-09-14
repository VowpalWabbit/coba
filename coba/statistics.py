from math import hypot, isnan, erf, sqrt
from statistics import fmean
from sys import version_info
from operator import mul, sub
from bisect import bisect_left
from itertools import repeat, accumulate, compress, chain
from abc import abstractmethod, ABC
from typing import Sequence, Tuple, Union, Callable, Optional, Literal

from coba.exceptions import CobaException
from coba.utilities import PackageChecker

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

class PointAndInterval(ABC):

    @abstractmethod
    def point(self, sample: Sequence[float]) -> float:
        ...

    @abstractmethod
    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        ...

class StdDevCI(PointAndInterval):
    def point(self,sample: Sequence[float]) -> float:
        return mean(sample)
    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        mu = mean(sample)
        sd = round(0 if len(sample) == 1 else stdev(sample),5)
        return (mu, (sd,sd))

class StdErrCI(PointAndInterval):

    def __init__(self, z_score:float=1.96) -> None:
        self._z_score = z_score
        self._inner   = StdDevCI()

    def point(self,sample: Sequence[float]) -> float:
        return self._inner.point(sample)

    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        (mu,(sd,sd)) = self._inner.point_interval(sample)
        sqrtn = len(sample)**.5
        se    = round(sd/sqrtn,5)
        ci    = self._z_score*se
        return (mu, (ci,ci))

class BootstrapCI(PointAndInterval):

    def __init__(self, confidence:float, statistic:Callable[[Sequence[float]], float]) -> None:
        PackageChecker.scipy('BootstrapConfidenceInterval')
        self._conf = confidence
        self._stat = statistic
        self._args = dict(method='basic', vectorized=False, n_resamples=1000, random_state=1)

    def point(self,sample: Sequence[float]) -> float:
        return self._stat(sample)

    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        from scipy.stats import bootstrap

        p   = self._stat(sample)

        if len(sample) < 3:
            l,h = p,p
        else:
            l,h = bootstrap([sample], self._stat, **self._args).confidence_interval

        return (p, (p-l,h-p))

class BinomialCI(PointAndInterval):

    def __init__(self, method:Literal['wilson', 'clopper-pearson']):
        self._method = method

    def point(self,sample: Sequence[float]) -> float:
        return sum(sample)/len(sample)

    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        if set(sample) - set([0,1]):
            raise CobaException("A binomial confidence interval can only be calculated on values of 0 and 1.")

        if self._method == "wilson":
            z_975 = 1.96 #z-score for .975 area to the left
            p_hat = sum(sample)/len(sample)
            n     = len(sample)
            Q     = z_975**2/(2*n)

            #https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
            interval_num = z_975*((p_hat*(1-p_hat))/n + Q/(2*n))**(.5)
            location_num = (p_hat+Q)

            interval_den = (1+2*Q)
            location_den = (1+2*Q)

            interval = interval_num/interval_den
            location = location_num/location_den

            return (p_hat, (p_hat-(location-interval), (location+interval)-p_hat))

        else:
            PackageChecker.scipy("BinomialConfidenceInterval")
            from scipy.stats import beta

            lo = beta.ppf(.05/2, sum(sample), len(sample) - sum(sample) + 1)
            hi = beta.ppf(1-.05/2, sum(sample) + 1, len(sample) - sum(sample))
            p_hat = sum(sample)/len(sample)

            lo = 0.0 if isnan(lo) else lo
            hi = 1.0 if isnan(hi) else hi

            return (p_hat, (p_hat-lo,hi-p_hat))

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
