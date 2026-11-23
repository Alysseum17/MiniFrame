import pytest

import miniframe as mf


def test_literal_window_uses_first_referenced_column_name() -> None:
    df = mf.DataFrame({"key": ["a", "a", "b"]})
    window = mf.lit(1).sum().over(mf.lit(True), "key")
    assert df.select(window).to_dicts() == [{"key": 2}, {"key": 2}, {"key": 1}]
    with pytest.raises(mf.SchemaError, match="^duplicate output name: 'key'$"):
        df.select("key", window)
    assert df.select(mf.lit(1).sum().over(mf.lit(True))).columns == ["literal"]
