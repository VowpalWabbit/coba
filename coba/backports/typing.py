import sys

if sys.version_info >= (3,8):# pragma: no cover
    from typing import Literal, Protocol
else:
    from typing_extensions import Literal, Protocol