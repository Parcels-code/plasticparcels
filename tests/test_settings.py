import plasticparcels as pp


def test_settings():
    settings = pp.utils.load_settings('tests/test_data/test_settings.json')
    assert settings is not None
