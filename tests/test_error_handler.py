"""error_handler.py 单元测试 — 重试/降级/边界"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from utils.error_handler import safe_execute, with_retry, streamlit_safe

class TestErrorHandler:
    def test_safe_execute_success(self):
        result, err = safe_execute(lambda x: x * 2, 21)
        assert result == 42
        assert err is None

    def test_safe_execute_returns_fallback(self):
        result, err = safe_execute(lambda: 1 / 0, fallback_value=99)
        assert result == 99
        assert err is not None

    def test_with_retry_success(self):
        call_count = [0]

        @with_retry(max_retries=3, backoff_factor=0.01, exceptions=(ValueError,))
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("temp")
            return "ok"

        assert flaky() == "ok"
        assert call_count[0] == 3

    def test_with_retry_exhausted(self):
        @with_retry(max_retries=2, backoff_factor=0.01, exceptions=(ValueError,))
        def always_fail():
            raise ValueError("always")

        with pytest.raises(RuntimeError, match="仍然失败"):
            always_fail()

    def test_safe_execute_with_none(self):
        result, err = safe_execute(lambda x: x, None, fallback_value=42)
        # lambda returns None which is fine, no exception
        assert result is None
        assert err is None
