from importlib.metadata import distribution


def test_installed_package_is_importable() -> None:
    import miniframe as mf

    assert mf.__name__ == distribution("miniframe").metadata["Name"]
