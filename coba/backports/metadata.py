import sys

if sys.version_info >= (3,10):#pragma: no cover
    from importlib.metadata import entry_points

elif sys.version_info >= (3,8):#pragma: no cover
    def entry_points(*,group):
        import importlib.metadata
        return importlib.metadata.entry_points()[group]

else:#pragma: no cover
    from importlib_metadata import entry_points
