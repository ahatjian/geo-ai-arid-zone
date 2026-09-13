"""结构化日志测试 — logging_config

该模块此前零引用、零覆盖。本轮接入到 llm / pc_data / error_handler 三处
真实的可观测性缺口, 这里把它的对外契约钉死。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib
import logging
import time
import pytest


@pytest.fixture(autouse=True)
def _capture_geo_logger(caplog):
    """捕获 geo-ai logger 的输出"""
    caplog.set_level(logging.INFO, logger="geo-ai")
    return caplog


class TestLogTimer:
    def test_records_operation_and_elapsed(self, caplog):
        from utils.logging_config import LogTimer
        with LogTimer("测试操作"):
            time.sleep(0.05)
        assert "[TIMER]" in caplog.text
        assert "测试操作" in caplog.text

    def test_elapsed_attribute_populated(self):
        from utils.logging_config import LogTimer
        with LogTimer("x") as timer:
            time.sleep(0.03)
        assert timer.elapsed >= 0.03

    def test_enter_returns_self(self):
        from utils.logging_config import LogTimer
        timer = LogTimer("x")
        with timer as returned:
            assert returned is timer
            assert returned.operation == "x"

    def test_still_logs_when_body_raises(self, caplog):
        """异常路径同样要留下耗时记录 —— 否则最需要排查的失败反而不计时"""
        from utils.logging_config import LogTimer
        with pytest.raises(ValueError):
            with LogTimer("失败操作"):
                raise ValueError("boom")
        assert "失败操作" in caplog.text


class TestLogFunctions:
    def test_log_info(self, caplog):
        from utils.logging_config import log_info
        log_info("模块A", "操作完成")
        assert "[INFO] 模块A: 操作完成" in caplog.text

    def test_log_warning(self, caplog):
        from utils.logging_config import log_warning
        log_warning("模块B", "请注意")
        assert "[WARN] 模块B: 请注意" in caplog.text

    def test_log_error_includes_exception_type_and_message(self, caplog):
        from utils.logging_config import log_error
        log_error("模块C", ValueError("坏值"))
        assert "[ERROR] 模块C" in caplog.text
        assert "ValueError" in caplog.text
        assert "坏值" in caplog.text

    def test_log_error_without_context(self, caplog):
        from utils.logging_config import log_error
        log_error("模块D", RuntimeError("x"))
        assert "模块D" in caplog.text

    def test_log_error_with_context(self, caplog):
        from utils.logging_config import log_error
        log_error("模块E", RuntimeError("x"), context={"step": "下载影像"})
        assert "下载影像" in caplog.text

    def test_log_api_call_success(self, caplog):
        from utils.logging_config import log_api_call
        log_api_call("DeepSeek", 1.234, success=True)
        assert "[API]" in caplog.text
        assert "DeepSeek" in caplog.text
        assert "1.23s" in caplog.text
        assert "✅" in caplog.text

    def test_log_api_call_failure_with_status(self, caplog):
        from utils.logging_config import log_api_call
        log_api_call("STAC", 5.0, success=False, status="超时降级")
        assert "❌" in caplog.text
        assert "超时降级" in caplog.text

    def test_log_api_call_without_status(self, caplog):
        from utils.logging_config import log_api_call
        log_api_call("X", 0.5, success=True)
        # 无 status 时不应出现多余的分隔符
        assert "[API] ✅ X | 0.50s" in caplog.text


class TestWithLogging:
    def test_preserves_return_value(self):
        from utils.logging_config import with_logging

        @with_logging("测试")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_preserves_function_metadata(self):
        """functools.wraps 应保留原函数名与 docstring (否则调试栈会全部显示 wrapper)"""
        from utils.logging_config import with_logging

        @with_logging("测试")
        def my_func():
            """我的文档"""

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "我的文档"

    def test_logs_qualified_name(self, caplog):
        from utils.logging_config import with_logging

        @with_logging("Mod")
        def slow_op():
            time.sleep(0.02)

        slow_op()
        assert "Mod.slow_op" in caplog.text

    def test_exception_propagates_unchanged(self):
        """装饰器不应吞掉异常"""
        from utils.logging_config import with_logging

        @with_logging("测试")
        def boom():
            raise KeyError("k")

        with pytest.raises(KeyError):
            boom()


class TestLoggerConfiguration:
    def test_logger_name_is_geo_ai(self):
        from utils.logging_config import _logger
        assert _logger.name == "geo-ai"

    def test_reload_does_not_stack_handlers(self):
        """重复加载模块不应叠加 handler, 否则每条日志会输出多次"""
        import utils.logging_config as lc
        before = len(lc._logger.handlers)
        importlib.reload(lc)
        assert len(lc._logger.handlers) == before


class TestIntegrationWithCallers:
    """三个接入点的实际行为"""

    def test_llm_records_api_call(self, caplog, monkeypatch):
        """query_deepseek 成功时应记录耗时与尝试次数"""
        import utils.llm as llm

        class _Resp:
            status_code = 200
            text = "ok"

            def json(self):
                return {"choices": [{"message": {"content": '{"modules": ["植被分析"]}'}}]}

        monkeypatch.setattr("requests.post", lambda *a, **kw: _Resp(), raising=False)
        llm.query_deepseek("分析植被", api_key="fake-key")
        assert "[API]" in caplog.text
        assert "DeepSeek" in caplog.text

    def test_error_boundary_records_structured_log(self, caplog):
        """无 Streamlit 上下文时, 错误应进入结构化日志而非仅 print"""
        from utils.error_handler import StreamlitErrorBoundary

        with StreamlitErrorBoundary("测试页", st=None):
            raise ValueError("边界内错误")

        assert "[ERROR]" in caplog.text
        assert "测试页" in caplog.text
        assert "ValueError" in caplog.text
