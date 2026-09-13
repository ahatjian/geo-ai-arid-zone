"""首页入口测试 — app.py

app.py 是 st.navigation 的入口文件, 承担三件事:
  1. 声明页面清单 (驱动侧边栏导航)
  2. 渲染全站共享侧边栏 (离线演示模式开关)
  3. 渲染首页面板

⚠️ 已知限制: streamlit.testing 的 AppTest.switch_page() **不会重跑入口脚本**,
只执行目标页面脚本, 因此它无法验证"共享 sidebar 在子页面上也存在" ——
那条性质只能靠真实浏览器回归验证。若在这里用它下结论, 会得到假阴性。
本文件只测入口自身的契约。
"""

import ast
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app.py')


def _load_page_specs():
    """用 AST 解析 _PAGE_SPECS。

    不 import app.py —— 那会执行 Streamlit 渲染代码并需要运行时。
    页面清单是纯字面量, AST 解析已经足够且更快。
    """
    with open(APP_PATH, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '_PAGE_SPECS':
                    return ast.literal_eval(node.value)
    raise AssertionError("app.py 中未找到 _PAGE_SPECS")


class TestPageSpecs:
    """页面清单是导航的数据源 —— 写错一个路径, 该页面就会从导航里消失。
    这类错误不会让应用报错, 只会让功能静默少一块, 所以用测试钉死。
    """

    def test_all_page_files_exist(self):
        specs = _load_page_specs()
        root = os.path.dirname(APP_PATH)
        missing = [p for _, p, _ in specs
                   if not os.path.exists(os.path.join(root, p))]
        assert not missing, f"以下页面文件不存在: {missing}"

    def test_page_count(self):
        assert len(_load_page_specs()) == 25

    def test_paths_unique(self):
        paths = [p for _, p, _ in _load_page_specs()]
        assert len(paths) == len(set(paths)), "页面路径重复"

    def test_titles_unique(self):
        """标题重复会让侧边栏导航出现两个同名条目, 用户无法区分"""
        titles = [t for t, _, _ in _load_page_specs()]
        assert len(titles) == len(set(titles)), "页面标题重复"

    def test_spec_shape(self):
        for spec in _load_page_specs():
            assert len(spec) == 3, f"页面项应为 (标题, 路径, 图标): {spec}"

    def test_pages_live_under_pages_dir(self):
        for _, path, _ in _load_page_specs():
            assert path.startswith("pages/"), f"页面路径应在 pages/ 下: {path}"
            assert path.endswith(".py"), f"页面应为 .py 文件: {path}"

    def test_every_page_has_icon(self):
        for title, _, icon in _load_page_specs():
            assert icon, f"{title} 缺少图标"

    def test_page_files_are_non_trivial(self):
        """页面文件不应是空壳 —— 防止误提交占位文件"""
        root = os.path.dirname(APP_PATH)
        for title, path, _ in _load_page_specs():
            full = os.path.join(root, path)
            size = os.path.getsize(full)
            assert size > 500, f"{title} ({path}) 仅 {size} 字节, 疑似占位文件"


class TestAppStructure:
    """app.py 的结构性约束 (静态检查, 无需运行)"""

    def test_uses_st_navigation(self):
        """必须走 st.navigation —— 这是全局 sidebar 共享机制的前提"""
        with open(APP_PATH, encoding='utf-8') as f:
            src = f.read()
        assert "st.navigation(" in src
        assert "st.Page(" in src

    def test_set_page_config_precedes_sidebar(self):
        """set_page_config 必须在任何 st 元素命令之前。

        页面自身的 set_page_config 会覆盖入口设置, 但入口那份若晚于
        sidebar 渲染, Streamlit 会报 "must be the first command"。
        """
        with open(APP_PATH, encoding='utf-8') as f:
            src = f.read()
        assert src.index("st.set_page_config(") < src.index("with st.sidebar:")

    def test_global_sidebar_renders_demo_toggle(self):
        with open(APP_PATH, encoding='utf-8') as f:
            src = f.read()
        assert "render_demo_mode_toggle" in src


class TestHomePageRendering:
    """真实运行入口脚本 (AppTest), 验证首页可渲染"""

    @pytest.fixture(scope="class")
    def at(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(APP_PATH, default_timeout=240)
        app.run()
        return app

    def test_no_exception(self, at):
        assert not at.exception, [str(e.value) for e in at.exception]

    def test_renders_platform_title(self, at):
        titles = [t.value for t in at.title]
        assert any("平台" in t for t in titles), f"未渲染平台标题: {titles}"

    def test_global_demo_toggle_present_in_sidebar(self, at):
        """全站共享的演示模式开关必须出现在侧边栏"""
        labels = [c.label for c in at.sidebar.checkbox]
        assert any("离线演示模式" in label for label in labels), f"侧边栏开关: {labels}"

    def test_sidebar_not_empty(self, at):
        """侧边栏应同时承载页面导航与共享开关"""
        assert at.sidebar is not None
        assert len(at.sidebar.checkbox) >= 1
