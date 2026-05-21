"""共享 fixtures"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest

@pytest.fixture
def mock_bands():
    np.random.seed(42)
    H, W = 50, 50
    return np.stack([
        0.08 + np.random.randn(H, W) * 0.01,
        0.12 + np.random.randn(H, W) * 0.02,
        0.15 + np.random.randn(H, W) * 0.02,
        0.30 + np.random.randn(H, W) * 0.03,
        0.25 + np.random.randn(H, W) * 0.03,
        0.22 + np.random.randn(H, W) * 0.02,
    ])

@pytest.fixture
def mock_ndvi_stack():
    np.random.seed(42)
    H, W, T = 40, 40, 24
    stack = np.zeros((H, W, T))
    for t in range(T):
        stack[:, :, t] = 0.3 + 0.15 * np.sin(2*np.pi*t/12) + np.random.randn(H, W)*0.03
    return stack
