"""离线演示模式开关测试 — demo_mode

开关状态存放在环境变量 GEOAI_DEMO_MODE, 是 UI (app.py 全局侧边栏) 与
底层模块 (pc_data / demo_data) 共用的唯一真源, 因此这里重点验证:
状态读写、与 st.checkbox 的同步、以及 pc_data 确实委托到同一真源。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例从"未设置"状态开始, 且不污染进程环境"""
    monkeypatch.delenv("GEOAI_DEMO_MODE", raising=False)


class TestIsDemoMode:
    def test_default_off(self):
        """未设置环境变量时默认关闭"""
        from utils.demo_mode import is_demo_mode
        assert is_demo_mode() is False

    def test_literal_one_enables(self, monkeypatch):
        monkeypatch.setenv("GEOAI_DEMO_MODE", "1")
        from utils.demo_mode import is_demo_mode
        assert is_demo_mode() is True

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("GEOAI_DEMO_MODE", "0")
        from utils.demo_mode import is_demo_mode
        assert is_demo_mode() is False

    @pytest.mark.parametrize("value", ["", "true", "yes", "TRUE", "True", "2", "on"])
    def test_only_literal_one_counts_as_enabled(self, monkeypatch, value):
        """只认 "1"。

        若用真值判断, 部署时写 GEOAI_DEMO_MODE=false 会被判为"已启用",
        导致线上静默切到合成数据 —— 这类误判不会报错, 只会让用户
        拿到假的影像。
        """
        monkeypatch.setenv("GEOAI_DEMO_MODE", value)
        from utils.demo_mode import is_demo_mode
        assert is_demo_mode() is False


class TestSetDemoMode:
    def test_enable_writes_one(self):
        from utils.demo_mode import set_demo_mode, is_demo_mode
        set_demo_mode(True)
        assert os.environ["GEOAI_DEMO_MODE"] == "1"
        assert is_demo_mode() is True

    def test_disable_writes_zero_rather_than_unset(self):
        """关闭时写 "0" 而不是删除变量, 状态始终可读"""
        from utils.demo_mode import set_demo_mode, is_demo_mode
        set_demo_mode(True)
        set_demo_mode(False)
        assert os.environ["GEOAI_DEMO_MODE"] == "0"
        assert is_demo_mode() is False

    def test_round_trip(self):
        from utils.demo_mode import set_demo_mode, is_demo_mode
        for state in (True, False, True, False):
            set_demo_mode(state)
            assert is_demo_mode() is state


class TestEnvName:
    def test_env_name_is_stable(self):
        """环境变量名是外部契约 (Dockerfile / 部署文档都引用它)"""
        from utils.demo_mode import DEMO_MODE_ENV
        assert DEMO_MODE_ENV == "GEOAI_DEMO_MODE"


class TestRenderToggle:
    """render_demo_mode_toggle 渲染 st.checkbox 并同步 env。

    st.checkbox 需要 Streamlit runtime, 这里替换掉该命令本身,
    专注验证"取值 → 同步"的契约。
    """

    def test_checkbox_on_enables_mode(self, monkeypatch):
        import streamlit as st
        monkeypatch.setattr(st, "checkbox", lambda *a, **kw: True)
        monkeypatch.setattr(st, "caption", lambda *a, **kw: None)

        from utils.demo_mode import render_demo_mode_toggle, is_demo_mode
        assert render_demo_mode_toggle() is True
        assert is_demo_mode() is True

    def test_checkbox_off_disables_mode(self, monkeypatch):
        """从开启状态点回关闭, 应同步 env (而非保留旧值)"""
        import streamlit as st
        monkeypatch.setattr(st, "checkbox", lambda *a, **kw: False)
        monkeypatch.setattr(st, "caption", lambda *a, **kw: None)

        from utils.demo_mode import render_demo_mode_toggle, is_demo_mode, set_demo_mode
        set_demo_mode(True)
        assert render_demo_mode_toggle() is False
        assert is_demo_mode() is False

    def test_caption_hidden_when_off(self, monkeypatch):
        import streamlit as st
        captured = []
        monkeypatch.setattr(st, "checkbox", lambda *a, **kw: False)
        monkeypatch.setattr(st, "caption", lambda *a, **kw: captured.append(a))

        from utils.demo_mode import render_demo_mode_toggle
        render_demo_mode_toggle()
        assert captured == []

    def test_caption_shown_when_on(self, monkeypatch):
        import streamlit as st
        captured = []
        monkeypatch.setattr(st, "checkbox", lambda *a, **kw: True)
        monkeypatch.setattr(st, "caption", lambda *a, **kw: captured.append(a))

        from utils.demo_mode import render_demo_mode_toggle
        render_demo_mode_toggle()
        assert len(captured) == 1

    def test_checkbox_reflects_current_state(self, monkeypatch):
        """开关初始值要回填当前状态 —— 否则页面切走再回来会显示成关闭"""
        import streamlit as st
        seen = {}

        def fake_checkbox(label, value=None, **kw):
            seen["value"] = value
            return value

        monkeypatch.setattr(st, "checkbox", fake_checkbox)
        monkeypatch.setattr(st, "caption", lambda *a, **kw: None)

        from utils.demo_mode import render_demo_mode_toggle, set_demo_mode
        set_demo_mode(True)
        render_demo_mode_toggle()
        assert seen["value"] is True

    def test_key_can_be_overridden(self, monkeypatch):
        """widget key 可覆盖 (入口文件用默认值, 测试/嵌套场景可自定义)"""
        import streamlit as st
        seen = {}

        def fake_checkbox(label, value=None, **kw):
            seen["key"] = kw.get("key")
            return False

        monkeypatch.setattr(st, "checkbox", fake_checkbox)
        monkeypatch.setattr(st, "caption", lambda *a, **kw: None)

        from utils.demo_mode import render_demo_mode_toggle
        render_demo_mode_toggle(key="custom_key")
        assert seen["key"] == "custom_key"


class TestPcDataIntegration:
    """pc_data 是演示模式的实际消费方, 必须与 UI 共用同一真源"""

    def test_pc_data_delegates_to_demo_mode(self):
        from utils import pc_data
        from utils.demo_mode import set_demo_mode

        set_demo_mode(False)
        assert pc_data._is_demo_mode() is False

        set_demo_mode(True)
        assert pc_data._is_demo_mode() is True

    def test_pc_data_reads_external_env(self, monkeypatch):
        """部署环境直接设的 env 也要被 pc_data 读到"""
        monkeypatch.setenv("GEOAI_DEMO_MODE", "1")
        from utils import pc_data
        assert pc_data._is_demo_mode() is True
