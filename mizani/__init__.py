from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version('mizani')
except PackageNotFoundError:
    # package is not installed
    pass
