from coba.backports import version, PackageNotFoundError

try:
    __version__ = version('coba') #option #5 on https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
except PackageNotFoundError:
    __version__ = "0.0.0"
