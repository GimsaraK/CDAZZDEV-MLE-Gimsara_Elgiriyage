"""Shared pytest setup: make the repository root importable and provide synthetic data."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SYNTHETIC_START_PRICE = 100.0
SYNTHETIC_DAILY_VOL = 0.015
SYNTHETIC_SEED = 42


def make_ohlcv(n_days: int, start: str = "2023-01-02", seed: int = SYNTHETIC_SEED) -> pd.DataFrame:
    """Random-walk OHLCV frame on business days, shaped like yfinance output."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_days, tz="America/New_York")
    returns = rng.normal(0.0005, SYNTHETIC_DAILY_VOL, n_days)
    close = SYNTHETIC_START_PRICE * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    open_ = (high + low) / 2
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close * 0.98,
            "Volume": rng.integers(1_000_000, 5_000_000, n_days),
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Keep retries (behaviour under test) but remove the sleep between attempts."""
    from task1_financial.src import config

    monkeypatch.setattr(config, "RETRY_BACKOFF_MIN_SECONDS", 0)
    monkeypatch.setattr(config, "RETRY_BACKOFF_MAX_SECONDS", 0)


@pytest.fixture
def ohlcv_3y() -> pd.DataFrame:
    return make_ohlcv(756)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
