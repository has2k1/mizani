import subprocess
import sys
import textwrap

# Block pandas, then use the parts of mizani that work with pandas
# objects when pandas is in use.
CODE = textwrap.dedent(
    """
    import sys
    from datetime import datetime, timedelta

    class BlockPandas:
        def find_spec(self, name, path=None, target=None):
            if name == "pandas" or name.startswith("pandas."):
                raise ModuleNotFoundError("pandas is blocked")

    sys.meta_path.insert(0, BlockPandas())

    import numpy as np

    import mizani.bounds as bounds
    import mizani.breaks as breaks
    import mizani.labels as labels
    import mizani.palettes as palettes
    import mizani.scale as scale
    import mizani.transforms as transforms

    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert list(bounds.rescale(x)[[0, 1, 3]]) == [0, 1 / 3, 1]
    assert np.isnan(bounds.censor(x, (1, 2))[3])
    assert list(breaks.breaks_extended()((0, 10))) == [0, 2.5, 5, 7.5, 10]
    assert labels.label_number()([1, 2.5]) == ["1.0", "2.5"]

    limits = scale.scale_discrete.train(["b", "a", None])
    assert limits[:2] == ["a", "b"] and np.isnan(limits[2])
    mapped = scale.scale_discrete.map(
        ["a", "b", None], palettes.hue_pal(), limits[:2], na_value="grey"
    )
    assert mapped[2] == "grey"

    # numpy turns NaN in a list of strings into the string "nan"
    assert scale.scale_discrete.train(["b", "a", np.nan], na_rm=True) == [
        "a",
        "b",
    ]
    mapped = scale.scale_discrete.map(
        ["a", "b", np.nan], palettes.hue_pal(), ["a", "b"], na_value="grey"
    )
    assert mapped[2] == "grey"

    dt = transforms.datetime_trans()
    values = dt.transform([datetime(2020, 1, 1), datetime(2021, 1, 1)])
    assert len(breaks.breaks_date_width("3 months")(dt.inverse(values)))

    td = transforms.timedelta_trans()
    values = td.transform([timedelta(days=1), timedelta(days=10)])
    assert td.inverse(values)[1] == timedelta(days=10)
    assert len(breaks.breaks_timedelta()((timedelta(0), timedelta(days=5))))

    assert "pandas" not in sys.modules
    """
)


def test_pandas_is_optional():
    result = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
