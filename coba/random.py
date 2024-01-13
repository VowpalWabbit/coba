"""Random number generation with deterministic generation according to a seed.

Remarks:
    This implementation guarantees coba reproducibility independent of other generators.
"""

import math
import time

from math import floor
from itertools import compress, accumulate, islice
from operator import mul,add
from typing import Optional, Iterable, Sequence, Any

class CobaRandom:
    """A random number generator."""

    def __init__(self, seed: Optional[float] = None) -> None:
        """Instantiate a CobaRandom generator.

        Args:
            seed: The seed to use when starting random number generation.

        Remarks:
            The values for a,c,m below are taken from L'ecuyer (1999). In this paper he notes that
            these values for LCG should have an overall period of m though the period of lower bits
            will be much shorter. A solution he offers to this problem is to use much larger m (e.g.,
            2**128) and then only use the the top n most significant digits. For now, we aren't doing that.

        References:
            L'ecuyer, Pierre. "Tables of linear congruential generators of different sizes
            and good lattice structure." Mathematics of Computation 68.225 (1999): 249-260.
        """

        if isinstance(seed,int) or (isinstance(seed,float) and seed.is_integer()):
            seed = int(seed)
        else:
            seed = int.from_bytes(str(seed or time.time()).encode('utf-8'),"big") % 2**20

        self._seed  = seed
        self._randu = self._next_uniform(116646453,seed,9,2**30)
        self._randg = self._next_gaussian()

    @property
    def seed(self) -> int:
        """Initial seed for random number generation."""
        return self._seed

    def random(self, min:float=0, max:float=1) -> float:
        """Generate a uniform random number in [`min`,`max`).

        Args:
            min: The minimum value for the random numbers.
            max: The maximum value for the random numbers.

        Returns:
            The generated random number in [`min`,`max`).
        """
        return min+(max-min)*next(self._randu)

    def randoms(self, n:int, min:float=0, max:float=1) -> Sequence[float]:
        """Generate `n` uniform random numbers in [`min`,`max`).

        Args:
            n: How many random numbers should be generated.
            min: The minimum value for the random numbers.
            max: The maximum value for the random numbers.

        Returns:
            The `n` generated random numbers in [`min`,`max`).
        """
        if n and not isinstance(n, int) or n < 0:
            raise ValueError("n must be an integer greater than or equal to 0")

        min  = float(min)
        diff = max-min
        out  = self._randu

        if diff != 1:
            out = map(diff.__mul__,out)
        if min != 0:
            out = map(min.__add__,out)

        return list(islice(out,n)) if n is not None else out

    def shuffle(self, items: Iterable[Any], inplace: bool = False) -> Sequence[Any]:
        """Shuffle the order of items in a sequence.

        Args:
            items: The items that are to be shuffled.
            inplace: Shuffle the items in their given container.

        Remarks:
            This is the Richard Durstenfeld's method popularized by Donald
            Knuth in The Art of Computer Programming. This algorithm is
            unbiased (i.e., all possible permutations are equally likely to
            occur).

        Returns:
            A new order of the original items.
        """

        l = items if inplace else list(items)

        n = len(l)
        if n < 2: return l

        #i goes from 0 to n-2
        #j is always i <= j < n
        for i,j in enumerate(map(add,range(n),map(floor,map(mul,range(n,1,-1),self._randu)))):
            l[i], l[j] = l[j], l[i]

        return l

    def randint(self, a:int, b:int) -> int:
        """Generate a uniform random integer in [a, b].

        Args:
            a: The inclusive lower bound for the random integer.
            b: The inclusive upper bound for the random integer.

        Returns:
            A random integer in [a,b].
        """

        return a+floor((b-a+1)*next(self._randu))

    def randints(self, n:int, a:int, b:int) -> Sequence[int]:
        """Generate `n` uniform random integers in [a, b].

        Args:
            n: The number of random integers to generate.
            a: The inclusive lower bound for the random integer.
            b: The inclusive upper bound for the random integer.

        Returns:
            A sequence of `n` random integers in [a,b].
        """
        b=b+1
        if a == 0:
            return [floor(b*r) for r in islice(self._randu,n)]
        else:
            r_range = b-a
            return [floor(r_range*r) + a for r in islice(self._randu,n)]

    def choice(self, seq: Sequence[Any], weights:Sequence[float] = None) -> Any:
        """Choose a random item from the given sequence.

        Args:
            seq: The sequence to pick randomly from.
            weights: The frequency which seq is selected.

        Returns:
            An item in seq.
        """
        if weights is None:
            return seq[int(len(seq)*next(self._randu))]

        else:
            tot = sum(weights)
            if tot == 0: raise ValueError("The sum of weights cannot be zero.")
            return next(compress(seq, map((next(self._randu)*tot).__le__, accumulate(weights))))

    def gauss(self, mu:float=0, sigma:float=1) -> float:
        """Generate a random number from N(mu,sigma).

        Args:
            mu: The expectation of the distribution we are drawing from.
            sigma: The standard deviation of the distribution we are drawing form.

        Returns:
            A random number drawn from N(mu,sigma).
        """
        return self.gausses(1, mu, sigma)[0]

    def gausses(self, n:int, mu:float=0, sigma:float=1) -> Sequence[float]:
        """Generate `n` independent random numbers from N(mu,sigma).

        Args:
            n: The number of random numbers to generate.
            mu: The expectation of the distribution we are drawing from.
            sigma: The standard deviation of the distribution we are drawing form.

        Returns:
            The `n` random numbers drawn from N(mu,sigma).
        """
        return [mu+sigma*g for g in islice(self._randg,n) ]

    def _next_uniform(self, a, s, c, m) -> Iterable[float]:
        """Generate uniform random numbers in [0,1).

        Random numbers are generated using a linear congruential generator.
        """
        m_1 = m-1
        while True:
            #when m is a power of 2
            #this is equal to modulo m
            s = (a * s + c) & (m_1)
            yield s/m

    def _next_gaussian(self) -> Iterable[float]:
        """Generate `n` gaussian random numbers in N(0,1).

        Random numbers are generated using the Box-Muller transform.
        """

        sqrt = math.sqrt
        log  = math.log
        pi   = math.pi
        cos  = math.cos
        sin  = math.sin

        while True:
            R = sqrt(-2*log(next(self._randu)))
            S = 2*pi*next(self._randu)
            yield R*cos(S)
            yield R*sin(S)

    def __reduce__(self):
        return (CobaRandom,(self._seed,))

_random = CobaRandom()

def seed(seed: Optional[float]) -> None:
    """Set the seed for module functions.

    Args:
        seed: The seed for generating random numbers.

    Remarks:
        Note, this seed does not affect random numbers generated by the standard library
    """

    global _random

    _random = CobaRandom(seed)

def random(min:float=0, max:float=1) -> float:
    """Generate a uniform random number in [`min`,`max`).

    Args:
        min: The minimum value for the random numbers.
        max: The maximum value for the random numbers.

    Returns:
        A uniform random number in [`min`,`max`).
    """
    return _random.random(min,max)

def randoms(n: int, min:float=0, max:float=1) -> Sequence[float]:
    """Generate `n` uniform random numbers in [`min`,`max`).

    Args:
        n: How many uniform random numbers should be generated.
        min: The minimum value for the random numbers.
        max: The maximum value for the random numbers.

    Returns:
        The `n` uniform random numbers in [`min`,`max`).
    """

    return _random.randoms(n,min,max)

def shuffle(items: Iterable[Any], inplace:bool=False) -> Sequence[Any]:
    """Shuffle the order of items in a sequence.

    Args:
        items: The items that are to be shuffled.
        inplace: Shuffle the items in their given container.

    Returns:
        A new order of the original items.
    """

    return _random.shuffle(items, inplace)

def randint(a:int, b:int) -> int:
    """Generate a uniform random integer in [a, b].

    Args:
        a: The inclusive lower bound for the random integer.
        b: The inclusive upper bound for the random integer.

    Returns:
        A random integer in [a,b].
    """

    return _random.randint(a,b)

def randints(n:int, a:int, b:int) -> Sequence[int]:
    """Generate `n` uniform random integers in [a, b].

    Args:
        n: The number of random integers to generate.
        a: The inclusive lower bound for the random integer.
        b: The inclusive upper bound for the random integer.

    Returns:
        A sequence of `n` random integers in [a,b].
    """

    return _random.randints(n,a,b)

def choice(seq: Sequence[Any], weights:Sequence[float]=None) -> Any:
    """Choose a random item from the given sequence.

    Args:
        seq: The sequence to pick randomly from.
        weights: The frequency which seq is selected.

    Returns:
        An item in seq.
    """

    return _random.choice(seq, weights)

def gauss(mu:float=0, sigma:float=1) -> float:
    """Generate a random number from N(mu,sigma).

    Returns:
        A random number drawn from N(mu,sigma).
    """

    return _random.gauss(mu,sigma)

def gausses(n:int, mu:float=0, sigma:float=1) -> Sequence[float]:
    """Generate `n` independent random numbers from N(mu,sigma).

    Args:
        n: How many random numbers should be generated
        mu: The expectation of the distribution we are drawing from.
        sigma: The standard deviation of the distribution we are drawing form.

    Returns:
        The `n` random numbers drawn from N(mu,sigma).
    """

    return _random.gausses(n,mu,sigma)
