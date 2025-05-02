from .. import palette
from .._palette import PaletteKind as pk

__all__ = (
    "category10",
    "category20",
    "category20b",
    "category20c",
    "observable10",
    "tableau10",
    "tableau20",
)

# credit: https://vega.github.io/vega/docs/schemes/#category10
category10 = palette(
    name="category10",
    swatches=[
        [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
        ]
    ],
    kind=pk.qualitative,
)

# credit: https://vega.github.io/vega/docs/schemes/#category20
category20 = palette(
    name="category20",
    swatches=[
        [
            (31, 119, 180),
            (174, 199, 232),
            (255, 127, 14),
            (255, 187, 120),
            (44, 160, 44),
            (152, 223, 138),
            (214, 39, 40),
            (255, 152, 150),
            (148, 103, 189),
            (197, 176, 213),
            (140, 86, 75),
            (196, 156, 148),
            (227, 119, 194),
            (247, 182, 210),
            (127, 127, 127),
            (199, 199, 199),
            (188, 189, 34),
            (219, 219, 141),
            (23, 190, 207),
            (158, 218, 229),
        ]
    ],
    kind=pk.qualitative,
)

# credit: https://vega.github.io/vega/docs/schemes/#category20b
category20b = palette(
    name="category20b",
    swatches=[
        [
            (57, 59, 121),
            (82, 84, 163),
            (107, 110, 207),
            (156, 158, 222),
            (99, 121, 57),
            (140, 162, 82),
            (181, 207, 107),
            (206, 219, 156),
            (140, 109, 49),
            (189, 158, 57),
            (231, 186, 82),
            (231, 203, 148),
            (132, 60, 57),
            (173, 73, 74),
            (214, 97, 107),
            (231, 150, 156),
            (123, 65, 115),
            (165, 81, 148),
            (206, 109, 189),
            (222, 158, 214),
        ]
    ],
    kind=pk.qualitative,
)

# credit: https://vega.github.io/vega/docs/schemes/#category20c
category20c = palette(
    name="category20c",
    swatches=[
        [
            (49, 130, 189),
            (107, 174, 214),
            (158, 202, 225),
            (198, 219, 239),
            (230, 85, 13),
            (253, 141, 60),
            (253, 174, 107),
            (253, 208, 162),
            (49, 163, 84),
            (116, 196, 118),
            (161, 217, 155),
            (199, 233, 192),
            (117, 107, 177),
            (158, 154, 200),
            (188, 189, 220),
            (218, 218, 235),
            (99, 99, 99),
            (150, 150, 150),
            (189, 189, 189),
            (217, 217, 217),
        ]
    ],
    kind=pk.qualitative,
)

# credit: https://vega.github.io/vega/docs/schemes/#observable10
observable10 = palette(
    name="observable10",
    swatches=[
        [
            (66, 105, 208),
            (239, 177, 24),
            (255, 114, 92),
            (108, 197, 176),
            (60, 169, 81),
            (255, 138, 183),
            (164, 99, 242),
            (151, 187, 245),
            (156, 107, 78),
            (148, 152, 160),
        ]
    ],
    kind=pk.qualitative,
)

# credit: https://vega.github.io/vega/docs/schemes/#tableau10
tableau10 = palette(
    name="tableau10",
    swatches=[
        [
            (76, 120, 168),
            (245, 133, 24),
            (228, 87, 86),
            (114, 183, 178),
            (84, 162, 75),
            (238, 202, 59),
            (178, 121, 162),
            (255, 157, 166),
            (157, 117, 93),
            (186, 176, 172),
        ]
    ],
    kind=pk.qualitative,
)

# credit: https://vega.github.io/vega/docs/schemes/#tableau20
tableau20 = palette(
    name="tableau20",
    swatches=[
        [
            (76, 120, 168),
            (158, 202, 233),
            (245, 133, 24),
            (255, 191, 121),
            (84, 162, 75),
            (136, 210, 122),
            (183, 154, 32),
            (242, 207, 91),
            (67, 152, 148),
            (131, 188, 182),
            (228, 87, 86),
            (255, 157, 152),
            (121, 112, 110),
            (186, 176, 172),
            (214, 113, 149),
            (252, 191, 210),
            (178, 121, 162),
            (214, 165, 201),
            (158, 118, 95),
            (216, 181, 165),
        ]
    ],
    kind=pk.qualitative,
)
