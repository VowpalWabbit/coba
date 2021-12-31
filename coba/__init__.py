from coba.backports import version, PackageNotFoundError

try:
    __version__ = version('coba') #option #5 on https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
except PackageNotFoundError: #pragma: no cover
    __version__ = "0.0.0"
