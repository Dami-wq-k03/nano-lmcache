from importlib import import_module


def test_package_is_importable() -> None:
    package = import_module("nano_lmcache")
    assert package.__name__ == "nano_lmcache"
