import sys

if sys.version_info >= (3,10):#pragma: no cover
    from importlib.metadata import entry_points, version, PackageNotFoundError

elif sys.version_info >= (3,8):# pragma: no cover
    from importlib.metadata import version, PackageNotFoundError

    def entry_points(*,group):
        import importlib.metadata 
        return importlib.metadata.entry_points()[group]
else:
    from importlib_metadata import entry_points, version, PackageNotFoundError