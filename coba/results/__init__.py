"""Efficient Result loading, slicing, and analyzing """

from coba.results.core import Table, Result, TransactionDecode, TransactionEncode, TransactionResult, moving_average, Missing
from coba.results.errors import PointAndInterval, StdDevCI, StdErrCI, BootstrapCI, BinomialCI
