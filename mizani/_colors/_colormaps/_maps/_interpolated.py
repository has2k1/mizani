from mizani._colors import ColorMapKind as cmk
from mizani._colors import InterpolatedMap

__all__ = (
    # diverging
    "bwr",
    "seismic",
    # miscellaneous
    "gist_rainbow",
    "terrain",
)


seismic = InterpolatedMap(
    colors=(
        (0.0, 0.0, 0.3),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    ),
    kind=cmk.diverging,
)

bwr = InterpolatedMap(
    colors=(
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    kind=cmk.diverging,
)


gist_rainbow = InterpolatedMap(
    colors=(
        (1.00, 0.00, 0.16),
        (1.00, 0.00, 0.00),
        (1.00, 1.00, 0.00),
        (0.00, 1.00, 0.00),
        (0.00, 1.00, 1.00),
        (0.00, 0.00, 1.00),
        (1.00, 0.00, 1.00),
        (1.00, 0.00, 0.75),
    ),
    values=(0.000, 0.030, 0.215, 0.400, 0.586, 0.770, 0.954, 1.000),
)

terrain = InterpolatedMap(
    colors=(
        (0.2, 0.2, 0.6),
        (0.0, 0.6, 1.0),
        (0.0, 0.8, 0.4),
        (1.0, 1.0, 0.6),
        (0.5, 0.36, 0.33),
        (1.0, 1.0, 1.0),
    ),
    values=(0.00, 0.15, 0.25, 0.50, 0.75, 1.00),
)
