from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from typing import Literal, TypeAlias

    BrewerMapType: TypeAlias = Literal[
        "Diverging",
        "Qualitative",
        "Sequential",
    ]

    BrewerMapTypeAlt: TypeAlias = Literal[
        "div",
        "qual",
        "seq",
        "diverging",
        "qualitative",
        "sequential",
    ]
