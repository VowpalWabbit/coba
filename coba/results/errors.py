from math import isnan
from abc import ABC, abstractmethod
from typing import Sequence,Tuple,Callable,Literal

from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.statistics import mean, stdev, phi

class PointAndInterval(ABC):
    """Calculate a point estimate and a confidence interval."""

    @abstractmethod
    def point(self, sample: Sequence[float]) -> float:
        """Calculate point estimate of a statistic.

        Args:
            sample: Sample to calculate a statistic for.

        returns:
            A point estimate of a statistic.
        """
        ...

    @abstractmethod
    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        """Calculate a point estimate and a confidence interval.

        Args:
            sample: Sample to calculate a statistic and its confidence interval.

        returns:
            A point estimate of a statistic along with a hi/lo confidence interval.
                The lo and hi values should be the size of the plotted error bars.
        """
        ...

class StdDevCI(PointAndInterval):
    """Calculate mean and standard deviation interval."""

    def point(self,sample: Sequence[float]) -> float:
        return mean(sample)

    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        mu = mean(sample)
        sd = round(0 if len(sample) == 1 else stdev(sample),5)
        return (mu, (sd,sd))

class StdErrCI(PointAndInterval):
    """Calculate mean and standard error interval."""

    def __init__(self, confidence:float = .95) -> None:
        """Instantiate a StdErrCI.

        Args:
            confidence: The desired confidence level of the interval. Should be in [0,1).
        """

        bot = 0
        top = 10

        cdf_curr   = phi((bot+top)/2)
        cdf_target = .5+confidence/2

        while abs(cdf_curr - cdf_target) > .0001:
            if cdf_curr < cdf_target:
                bot = (bot+top)/2
            else:
                top = (bot+top)/2
            cdf_curr = phi((bot+top)/2)

        self._z_score = (bot+top)/2
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
    """Calculate a statistic and its Bootstrap CI."""

    def __init__(self, confidence:float, statistic:Callable[[Sequence[float]], float]) -> None:
        """Instantiate a BootstrapCI.

        Args:
            confidence: The desired confidence level of the interval. Should be in [0,1).
            statistic: A callable which returns a point estimate given a sample.
        """
        PackageChecker.scipy('BootstrapCI')
        self._stat = statistic
        self._args = dict(method='basic', vectorized=False, n_resamples=1000, random_state=1, confidence_level=confidence)

    def point(self,sample: Sequence[float]) -> float:
        return self._stat(sample)

    def point_interval(self, sample: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
        from scipy.stats import bootstrap

        p   = self._stat(sample)

        if len(sample) < 3:
            l,h = p,p
        else:
            l,h = bootstrap([sample], self._stat, **self._args).confidence_interval

        return (p, (max(p-l,0),max(h-p,0)))

class BinomialCI(PointAndInterval):
    """Calculate the mean and interval of a binomial."""

    def __init__(self, method:Literal['wilson', 'clopper-pearson']):
        """Instantiate a BinomialCI.

        Args:
            method: The method to calculate the Binomial confidence interval
        """
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

            lo = p_hat-(location-interval)
            hi = (location+interval)-p_hat

        else:
            PackageChecker.scipy("BinomialConfidenceInterval")
            from scipy.stats import beta

            lo = beta.ppf(.05/2, sum(sample), len(sample) - sum(sample) + 1)
            hi = beta.ppf(1-.05/2, sum(sample) + 1, len(sample) - sum(sample))
            p_hat = sum(sample)/len(sample)

            lo = 0.0 if isnan(lo) else lo
            hi = 1.0 if isnan(hi) else hi

            lo = p_hat-lo
            hi = hi-p_hat

        return (p_hat, (round(lo,5), round(hi,5)) )
