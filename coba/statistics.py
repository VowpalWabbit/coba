"""The statistics module contains algorithms and methods to calculate statistics."""

import collections

from math import isnan, sqrt, isclose, trunc, ceil, floor
from statistics import mean, variance
from numbers import Real, Rational, Complex
from typing import Sequence, Union, Dict, Any, overload, cast

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

class StatisticalEstimate(Rational):
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
        we won't have an estimate of their standard error. Ideally we won't be dealing with estimates from samples 
        of N==1, but in the case that we the NaN transfers through algebras. Attempts at determining an appropriate
        default always caused unintended consequences.

        WARNING!!! The algebra of random variables implemented below assumes every StatisticalEstimate is independent
        WARNING!!! The algebra of random variables implemented below assumes every StatisticalEstimate is independent
        WARNING!!! The algebra of random variables implemented below assumes every StatisticalEstimate is independent

        For more information on creating numerical types in Python see https://docs.python.org/3.6/library/numbers.html 
    """

    @overload
    def __init__(self, estimate: 'StatisticalEstimate') -> None:
        ...

    @overload
    def __init__(self, estimate: Rational, standard_error: float = float('nan')) -> None:
        ...

    @overload
    def __init__(self, estimate:float, standard_error: float = float('nan')) -> None:
        ...

    def __init__(self, estimate:Union['StatisticalEstimate', Rational, float], standard_error: float = float('nan')) -> None:
        
        if isinstance(estimate, Rational):
            unpacked_estimate = cast(Union[StatisticalEstimate, float], estimate.numerator/estimate.denominator)
        else:
            unpacked_estimate = estimate

        if isinstance(unpacked_estimate, StatisticalEstimate):
            assert isnan(standard_error) or standard_error == unpacked_estimate.standard_error, "Conflicting standard_errors were given"
            self._estimate       = unpacked_estimate.estimate
            self._standard_error = unpacked_estimate.standard_error
        else:
            self._estimate       = unpacked_estimate
            self._standard_error = standard_error

    @property
    def estimate(self) -> float:
        return self._estimate

    @property
    def standard_error(self) -> float:
        return self._standard_error

    #Region: Rational interface
    @property
    def numerator(self) -> 'StatisticalEstimate': #type: ignore
        #mypy doesn't like this. In theory typeshed says super().numerator should return an int.
        #the only way we can really satisfy that is if we define a `class IntegerEstimate(int)`
        #and return that here instead of `StatisticalEstimate` and maybe rename StatisticalEstimate
        #in that case to FloatEstimate.
        
        if isinstance(self.estimate, int):
            n,d = self.estimate, 1
        else:
            n,d = self.estimate.as_integer_ratio()
        
        return StatisticalEstimate(n, d*self.standard_error)

    @property
    def denominator(self) -> int:
        if isinstance(self.estimate, int):
            return 1
        else:
            return self.estimate.as_integer_ratio()[1]

    def __bool__(self) -> bool:
        return True

    def __complex__(self) -> 'complex':
        return complex(self.estimate)

    def __int__(self) -> int:
        return int(self.estimate)

    def __float__(self) -> float:
        return float(self.estimate)

    def __trunc__(self) -> int:
        return trunc(self.estimate)

    def __floor__(self) -> int:
        return floor(self.estimate)

    def __ceil__(self) -> int:
        return ceil(self.estimate)

    def __abs__(self) -> float:
        return abs(self.estimate)

    def __round__(self, ndigits=None) -> Any:
        return round(self.estimate, ndigits)

    def __hash__(self):
        return hash((self._estimate, self._standard_error))

    def __neg__(self) -> 'StatisticalEstimate':
        return StatisticalEstimate(-self._estimate, self._standard_error)

    def __pos__(self) -> 'StatisticalEstimate':
        return self

    def __add__(self, other: Any) -> 'StatisticalEstimate':
        if isinstance(other, StatisticalEstimate):
            estimate       = self.estimate+other.estimate
            standard_error = sqrt(self.standard_error**2+other.standard_error**2)
            return StatisticalEstimate(estimate,standard_error)

        if isinstance(other, Real):
            estimate       = self.estimate + other
            standard_error = self.standard_error
            return StatisticalEstimate(estimate,standard_error)

        return NotImplemented

    def __radd__(self, other: Any) -> 'StatisticalEstimate':
        return self + other

    def __sub__(self, other: Any) -> 'StatisticalEstimate':
        if isinstance(other, Complex):
            return self + (-other)

        return NotImplemented

    def __rsub__(self, other: Any) -> 'StatisticalEstimate':
        return (-self) + other

    def __mul__(self, other: Any) -> 'StatisticalEstimate':
        if isinstance(other, StatisticalEstimate):
            raise TypeError("We do not currently support multiplication of StatisticalEstimate by StatisticalEstimate.")

        if isinstance(other, Real):
            return StatisticalEstimate(other*self.estimate, other*self.standard_error)

        return NotImplemented

    def __rmul__(self, other: Any) -> 'StatisticalEstimate':
        return self * other

    def __truediv__(self, other: Any) -> 'StatisticalEstimate':
        if isinstance(other, StatisticalEstimate):
            raise TypeError("We do not currently support division of StatisticalEstimate by StatisticalEstimate.")

        if isinstance(other, Real):
            return self * (1/other)

        return NotImplemented

    def __rtruediv__(self, other: Any) -> 'StatisticalEstimate':
        raise TypeError("We do not currently support division by StatisticalEstimate.")

    def __floordiv__(self, other: Any) -> Union['StatisticalEstimate',int]:
        if self.estimate % other == 0:
            #in this case floordiv is being used to protect against floating point
            #errors and not to actually change the distribution of the random variable
            new_estimate = self/other
            return StatisticalEstimate(floor(new_estimate._estimate), new_estimate._standard_error)
        else:
            return floor(self/other)            

    def __rfloordiv__(self, other: Any) -> int:
        return floor(other/self)

    def __mod__(self, other: Any) -> float:
        if isinstance(other, StatisticalEstimate):
            raise TypeError("We do not currently support modulo of StatisticalEstimate by StatisticalEstimate.")

        if isinstance(other, Real):
            return int(self.estimate % other)

        return NotImplemented

    def __rmod__(self, other: Any) -> float:
        raise TypeError("We do not currently support modulo by StatisticalEstimate.")

    def __pow__(self, exponent: Any) -> Any:
        raise TypeError("We do not currently support multiplication of StatisticalEstimate by StatisticalEstimate.")

    def __rpow__(self, base: Any) -> Any:
        raise TypeError("We do not currently support exponentiation by StatisticalEstimate.")

    def __lt__(self, other: Any) -> bool:
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        return NotImplemented

    def __eq__(self, other) -> bool:

        eq = lambda a,b: isclose(a,b) or (isnan(a) and isnan(b))

        return isinstance(other, StatisticalEstimate) and eq(self.estimate,other.estimate) and eq(self.standard_error,other.standard_error)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __gt__(self, other: Any) -> bool:
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        return NotImplemented 
    #Region: Rational Interface

    @staticmethod
    def __from_json__(json:Dict[str,Any]) -> 'StatisticalEstimate':
        return StatisticalEstimate(json['estimate'], json['standard_error'])

    def __to_json__(self) -> Dict[str,Any]:
        return {
            'estimate'      : self._estimate,
            'standard_error': self._standard_error
        }

    def __str__(self) -> str:
        return str({'Est': round(self._estimate,4), 'SE': round(self._standard_error,4)})

    def __repr__(self) -> str:
        return str(self)

class BatchMeanEstimator(StatisticalEstimate):
    """Estimate the population mean from a batch of i.i.d. observations"""

    @overload
    def __init__(self, given: StatisticalEstimate) -> None:
        ...

    @overload
    def __init__(self, given: Sequence[float]) -> None:
        ...

    def __init__(self, given) -> None:
        if isinstance(given, collections.Sequence):
            estimate       = mean(given) if len(given) > 0 else float('nan')
            standard_error = sqrt(variance(given)/len(given)) if len(given) > 1 else float('nan')
            super().__init__(estimate, standard_error)
        else:
            super().__init__(given)