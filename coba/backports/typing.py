import sys

if sys.version_info >= (3,8):# pragma: no cover
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3,10):# pragma: no cover
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
