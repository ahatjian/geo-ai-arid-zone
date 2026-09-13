"""Tests for the AI-led Geo AI copilot."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import ai_copilot  # noqa: E402


def _synthetic_bands():
    h, w = 80, 80
    bands = np.zeros((6, h, w), dtype=np.float32)
    bands[0] = 0.08
    bands[1] = 0.14
    bands[2] = 0.12
    bands[3] = 0.30
    bands[4] = 0.25
    bands[5] = 0.22
    return bands


def test_generate_research_plan_with_fallback(monkeypatch):
    from utils.llm import fallback_parse

    monkeypatch.setattr(ai_copilot, "query_deepseek", lambda prompt, api_key=None: fallback_parse(prompt))
    plan = ai_copilot.generate_research_plan("分析植被和干旱风险")
    assert plan["study_area"]
    assert any("植被" in step["module"] for step in plan["steps"])
    assert any("干旱" in step["module"] for step in plan["steps"])
    assert plan["steps"]


def test_quick_scan_returns_core_metrics():
    scan = ai_copilot.quick_scan(_synthetic_bands(), modules=["植被分析", "干旱监测"])
    assert "ndvi_mean" in scan
    assert "vegetation_ratio" in scan
    assert "water_ratio" in scan
    assert "nddi_mean" in scan
    assert "drought_risk_ratio" in scan


def test_quick_scan_salinity_metrics():
    scan = ai_copilot.quick_scan(_synthetic_bands(), modules=["土壤盐渍化"])
    assert "ndsi_salinity_mean" in scan
    assert "salinity_ratio" in scan
