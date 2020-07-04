"""A custom implementation of random number generation.

Remarks:
    This implementation has been made to guarantee the reproducibility of benchmark tests
    according to integer seeds across all versions of Python. The standard implementation 
    of random within Python has had a few variations in implementation in the past and 
    could always change in the future, making randomization by seed potentially non-fixed.
"""

import math
import random as std_random

from typing import Optional, Iterator, Sequence, Union, Any, List, MutableSequence

class Random:
    """A random number generator via a linear congruential generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Instantiate a Random class.

        Args:
            seed: the seed to start random number generation.

        Remarks:
            The values for a,c,m below are taken from L’ecuyer (1999). In this paper he notes that
            these values for LCG should have an overall period of m though the period of lower bits 
            will be much shorter. A solution he offers to this problem is to use much larger m (e.g., 
            2**128) and then only use the the top n most significant digits. For now, we aren't doing that.
        
        References:
            L’ecuyer, Pierre. "Tables of linear congruential generators of different sizes 
            and good lattice structure." Mathematics of Computation 68.225 (1999): 249-260.
        """

        self._m = 2**30
        self._a = 116646453
        self._c = 9

        self._m_is_power_of_2 = math.log2(self._m).is_integer()
        self._m_minus_1       = self._m-1

        self._seed: int = std_random.randint(0,self._m_minus_1) if seed is None else seed

    def randoms(self, n:int=1) -> Sequence[float]:

        return [number/self._m_minus_1 for number in self._next(n)]

    def shuffle(self, array_like: MutableSequence[Any]) -> Sequence[Any]:

        n = len(array_like)
        r = self.randoms(n)

        for i in range(n):
            j = min(int(i + (r[i] * (n-i))), n-1) #min() handles the edge case of r[i]==1
            
            array_like[i], array_like[j] = array_like[j], array_like[i]


        return array_like

    def _next(self, n: int) -> Sequence[int]:
        """Linear congruential generator."""
        
        if n < 0 or not isinstance(n, int):
            raise ValueError("n must be an integer greater than 0")

        numbers: List[int] = []

        for _ in range(n):

            if self._m_is_power_of_2:
                self._seed = int((self._a * self._seed + self._c) & (self._m_minus_1))
            else:
                self._seed = int((self._a * self._seed + self._c) % self._m)

            numbers.append(self._seed)

        return numbers

_random = Random()

def seed(seed: Optional[int]) -> None:
    global _random
    
    _random = Random(seed)

def randoms(n: int) -> Sequence[float]:
    return _random.randoms(n)

def shuffle(array_like: MutableSequence[Any]) -> Sequence[Any]:
    return _random.shuffle(array_like)