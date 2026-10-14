import pytest


@pytest.fixture(scope="session")
def gri30():
    from autokinetics.library import load_example

    return load_example("methane_air_gri30")[0]


@pytest.fixture(scope="session")
def h2o2():
    from autokinetics.library import load_example

    return load_example("hydrogen_oxygen_ignition")[0]


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOKINETICS_CACHE_DIR", str(tmp_path / "cache"))
