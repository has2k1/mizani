"""
Check that the documentation reference lists each module's exports

`docs/great-docs.yml` names every documented object by hand, because
great-docs does not read the `__all__` of a submodule. A name added to
`__all__` but not to the reference would be missing from the website.
"""

import importlib
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent.parent / "docs" / "great-docs.yml"

# The modules whose exports make up the reference
MODULES = ("bounds", "breaks", "labels", "palettes", "transforms", "scale")


def listed_names(config: Path) -> set[str]:
    """
    Return the names in the reference sections of a great-docs configuration

    Parameters
    ----------
    config
        Path to the configuration file.
    """
    data = yaml.safe_load(config.read_text())
    return {
        item if isinstance(item, str) else item["name"]
        for section in data["reference"]["sections"]
        for item in section["contents"]
    }


def exported_names() -> set[str]:
    """
    Return the exports of the documented modules, as `module.name`
    """
    return {
        f"{module}.{name}"
        for module in MODULES
        for name in importlib.import_module(f"mizani.{module}").__all__
    }


def main() -> int:
    """
    Report the names that are in only one of the reference and `__all__`

    Returns
    -------
    :
        The exit status, 1 if any name is missing or stale and 0 otherwise.
    """
    listed = listed_names(CONFIG)
    exported = exported_names()
    missing = sorted(exported - listed)
    stale = sorted(listed - exported)
    for name in missing:
        print(f"Missing from the reference: {name}")
    for name in stale:
        print(f"In the reference but not exported: {name}")
    return 1 if missing or stale else 0


if __name__ == "__main__":
    sys.exit(main())
