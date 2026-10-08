"""Shared pytest setup: make the repository root importable and provide synthetic data."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# tests/ -> task1_financial/ -> repo root; added to sys.path so `import task1_financial.src...` works.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Recorded API responses (yfinance JSON, Google News XML) so tests never need the network.
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Synthetic price settings: start at 100, ~1.5% daily volatility, fixed seed for reproducible tests.
SYNTHETIC_START_PRICE = 100.0
SYNTHETIC_DAILY_VOL = 0.015
SYNTHETIC_SEED = 42


def make_ohlcv(n_days: int, start: str = "2023-01-02", seed: int = SYNTHETIC_SEED) -> pd.DataFrame:
    """Random-walk OHLCV frame on business days, shaped like yfinance output."""
    rng = np.random.default_rng(seed)
    # Business days with a New York timezone, like yfinance (the cleaner must strip the timezone).
    index = pd.bdate_range(start=start, periods=n_days, tz="America/New_York")
    # Geometric random walk: normally distributed daily log-returns with a slight upward drift.
    returns = rng.normal(0.0005, SYNTHETIC_DAILY_VOL, n_days)
    close = SYNTHETIC_START_PRICE * np.exp(np.cumsum(returns))
    # High is always above Close and Low always below it, so the bars are internally consistent.
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    open_ = (high + low) / 2
    # Adj Close is a fixed 2% below Close, so tests can tell raw and adjusted prices apart.
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
    """About 3 years of synthetic daily bars (252 trading days per year x 3)."""
    return make_ohlcv(756)


@pytest.fixture
def fixtures_dir() -> Path:
    """Folder with the recorded API responses."""
    return FIXTURES_DIR
