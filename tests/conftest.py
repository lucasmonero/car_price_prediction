import pytest

from car_price.config import load_config, resolve
from car_price.data import TARGET, clean, load_raw


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def clean_df(cfg):
    return clean(load_raw(resolve(cfg, "sample")), cfg)


@pytest.fixture(scope="session")
def Xy(clean_df):
    return clean_df.drop(columns=TARGET), clean_df[TARGET]
