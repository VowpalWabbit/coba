import sys

if sys.version_info >= (3,8):# pragma: no cover
    from importlib.metadata import entry_points, version
else:
    from importlib_metadata import entry_points, version 

__all__ = [
    'entry_points',
    'version'
]