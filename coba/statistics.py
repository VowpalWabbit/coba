from typing import Sequence, Tuple, Union

def iqr(values: Sequence[float]) -> float:

    if len(values) <= 1: return 0.

    values = sorted(values)

    p25,p75 = percentile(values, [0.25,0.75])

    return p75-p25

def percentile(values: Sequence[float], percentiles: Union[float,Sequence[float]]) -> Union[float, Tuple[float,...]]:

    def _percentile(values: Sequence[float], percentile: float) -> float:
        assert 0 <= percentile and percentile <= 1, "Percentile must be between 0 and 1 inclusive."

        i = percentile*(len(values)-1)

        if i == int(i):
            return values[int(i)]
        else:
            return values[int(i)] * (1-(i-int(i))) + values[int(i)+1] * (i-int(i))

    values = sorted(values)

    if isinstance(percentiles,(float,int)):
        return _percentile(values, percentiles)
    else:
        return tuple([_percentile(values, p) for p in percentiles ])

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
