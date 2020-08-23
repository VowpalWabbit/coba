"""The statistics module contains algorithms and methods to calculate statistics.

TODO Add unit tests to make sure StatisticalEstimate algebra works correctly
"""

from math import isnan, sqrt
from typing import Sequence, Union, Dict, Any
from statistics import mean, variance

from coba.json import JsonSerializable

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

class StatisticalEstimate(JsonSerializable):
    """An estimate of some statistic of interst along with useful additional statistics of that estimate.

    Remarks:
        In theory, if every StatisticalEstimate kept their entire sample then we could estimate values
        empirically by mixing all samples into a single big pot and then recalculating mean, and SEM.
        Unfortunately, this approach while simple has two drawbacks. First, COBA benchmarks would have 
        memory memory complexity of O(n) where n is the number of interactions in the benchmark. Second, 
        COBA would also have a computation complexity of O(n) everytime benchmark results are agreggated
        since aggregation would require reiterating over all interaction rewards. There are downsides to not
        taking the simple approach but the reduction in memory and compute complexity seems to be worth it. 

        One notable downside of not taking the simple approach is that stats with N == 1 can be problematic since 
        we won't have an estimate of their standard error. Below we handle this by using any standard error that is
        available to us as a stand-in (see __add__). Ideally we won't be dealing with estimates from samples of N==1.
    """

    def __init__(self, estimate:float, standard_error: float) -> None:
        self._estimate       = estimate
        self._standard_error = standard_error

    @property
    def estimate(self) -> float:
        return self._estimate

    @property
    def standard_error(self) -> float:
        return self._standard_error

    def __add__(self, other) -> 'StatisticalEstimate':
        if isinstance(other, (int,float)):
            return StatisticalEstimate(other+self.estimate, self.standard_error)

        if isinstance(other, StatisticalEstimate):
            if isnan(self.standard_error):
                #since we don't know our own SE use other as a best guess...
                standard_error = sqrt(other.standard_error**2 + other.standard_error**2)
            elif isnan(other.standard_error):
                #since we don't know other's SE use our own as a best guess...
                standard_error = sqrt(self.standard_error**2 + self.standard_error**2)
            else:
                standard_error = sqrt(self.standard_error**2+other.standard_error**2)

            return StatisticalEstimate(self.estimate+other.estimate, standard_error)

        raise Exception(f"Unable to add StatisticalEstimate and {type(other).__name__}")

    def __radd__(self, other) -> 'StatisticalEstimate':
        return self + other

    def __sub__(self, other) -> 'StatisticalEstimate':
        return self + (-other)

    def __rsub__(self, other) -> 'StatisticalEstimate':
        return (-self) + other

    def __mul__(self, other) -> 'StatisticalEstimate':
        if isinstance(other, (int,float)):
            return StatisticalEstimate(other*self.estimate, other*self.standard_error)
        
        if isinstance(other, StatisticalEstimate):
            raise Exception("We do not currently support multiplication by StatisticalEstimate.")

        raise Exception(f"Unable to multiply StatisticalEstimate and {type(other).__name__}")

    def __rmul__(self, other) -> 'StatisticalEstimate':
        return self * other

    def __truediv__(self,other) -> 'StatisticalEstimate':
        if isinstance(other, (int,float)):
            return self * (1/other)

        if isinstance(other, StatisticalEstimate):
            raise Exception("We do not currently support division by StatisticalEstimate.")

        raise Exception(f"Unable to divide StatisticalEstimate and {type(other).__name__}")

    def __rtruediv__(self,other) -> 'StatisticalEstimate':
        return self/other

    def __neg__(self) -> 'StatisticalEstimate':
        return -1 * self

    def __eq__(self, other) -> bool:
        
        eq = lambda a,b: (a == b) or (isnan(a) and isnan(b))
        
        return isinstance(other, StatisticalEstimate) and eq(self.estimate,other.estimate) and eq(self.standard_error,other.standard_error)

    @staticmethod
    def __from_json_obj__(json:Dict[str,Any]) -> 'StatisticalEstimate':
        return StatisticalEstimate(json['estimate'], json['standard_error'])

    def __to_json_obj__(self) -> Dict[str,Any]:
        return {
            '_type'         : 'StatisticalEstimate',
            'estimate'      : self._estimate,
            'standard_error': self._standard_error
        }

    def __str__(self) -> str:
        return str({'Est': round(self._estimate,4), 'SE': round(self._standard_error,4)})

    def __repr__(self) -> str:
        return str(self)

class BatchMeanEstimator(StatisticalEstimate):
    """Estimate the population mean from a batch of i.i.d. observations"""

    def __init__(self, sample: Sequence[float]) -> None:
        estimate       = mean(sample) if len(sample) > 0 else float('nan')
        standard_error = sqrt(variance(sample)/len(sample)) if len(sample) > 1 else float('nan')

        super().__init__(estimate, standard_error)

def coba_mean(values:Sequence[Union[StatisticalEstimate,float]]) -> Union[StatisticalEstimate,float]:
    return sum(values) / len(values)

def coba_weighted_mean(weights:Sequence[float], values:Sequence[Union[StatisticalEstimate,float]]) -> Union[StatisticalEstimate,float]:
    return sum([weight * value for weight,value in zip(weights,values) ]) / sum(weights)