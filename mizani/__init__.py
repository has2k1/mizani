from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version('plotnine')
except PackageNotFoundError:
    # package is not installed
    pass
