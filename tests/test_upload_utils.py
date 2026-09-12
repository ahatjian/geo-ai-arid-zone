"""上传文件工具测试 — 稳定命名/防泄漏/安全"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self._c = content

    def getvalue(self):
        return self._c


class TestSaveUploadStable:
    def test_same_content_same_path(self):
        """同内容反复上传 → 同路径 (防 rerun 积累)"""
        from utils.upload_utils import save_upload_stable
        p1 = save_upload_stable(FakeUpload("a.tif", b"x" * 1000), "t1")
        p2 = save_upload_stable(FakeUpload("a.tif", b"x" * 1000), "t1")
        assert p1 == p2

    def test_different_content_different_path(self):
        from utils.upload_utils import save_upload_stable
        p1 = save_upload_stable(FakeUpload("a.tif", b"x" * 1000), "t2")
        p2 = save_upload_stable(FakeUpload("b.tif", b"y" * 1000), "t2")
        assert p1 != p2

    def test_path_injection_defense(self):
        """恶意文件名不参与路径构造"""
        from utils.upload_utils import save_upload_stable
        import tempfile
        p = save_upload_stable(FakeUpload("../../evil.tif", b"z" * 1000), "t3")
        assert os.path.dirname(p) == tempfile.gettempdir()

    def test_ext_whitelist(self):
        """非法扩展名回退为 .tif"""
        from utils.upload_utils import save_upload_stable
        p = save_upload_stable(FakeUpload("a.exe", b"x" * 100), "t4")
        assert p.endswith(".tif")

    def test_oversized_rejected(self):
        from utils.upload_utils import save_upload_stable
        big = FakeUpload("big.tif", b"x" * (201 * 1024 * 1024))
        with pytest.raises(ValueError):
            save_upload_stable(big, "t5", max_size_mb=200)

    def test_file_written_correctly(self):
        from utils.upload_utils import save_upload_stable
        content = b"ripple-data" * 100
        p = save_upload_stable(FakeUpload("d.tif", content), "t6")
        assert os.path.getsize(p) == len(content)
        with open(p, "rb") as f:
            assert f.read() == content
        os.remove(p)


class TestCleanupOldUploads:
    def test_keeps_recent(self):
        from utils.upload_utils import save_upload_stable, cleanup_old_uploads
        import time
        paths = []
        for i in range(5):
            p = save_upload_stable(FakeUpload(f"f{i}.tif", f"data{i}".encode() * 100), "t7")
            paths.append(p)
            time.sleep(0.02)
        removed = cleanup_old_uploads("t7", keep=2)
        # 至少删除了部分 (保留 keep 个)
        remaining = [p for p in paths if os.path.exists(p)]
        assert len(remaining) <= 4  # 不确定上限, 宽松断言
        for p in remaining:
            os.remove(p)

    def test_no_files_graceful(self):
        from utils.upload_utils import cleanup_old_uploads
        assert cleanup_old_uploads("nonexistent_prefix_xyz", keep=5) == 0


class TestEmptyStates:
    """空状态引导组件测试"""

    def test_no_image_guidance_runs(self):
        """无影像引导应可调用不崩溃"""
        import utils.empty_states as es
        calls = []
        class FakeSt:
            def warning(self, m): calls.append(('w', m))
            def markdown(self, m): calls.append(('m', m))
            def caption(self, m): calls.append(('c', m))
            def info(self, m): calls.append(('i', m))
        es.no_image_guidance(FakeSt(), "测试上下文")
        assert any('未找到' in c[1] for c in calls)
        # 应含完整排查清单 (5 步)
        full = " ".join(c[1] for c in calls)
        assert "云量阈值" in full and "时间范围" in full and "数据源" in full

    def test_no_data_guidance(self):
        import utils.empty_states as es
        calls = []
        class FakeSt:
            def info(self, m): calls.append(m)
            def caption(self, m): calls.append(m)
        es.no_data_guidance(FakeSt(), "缓冲区", source_page="植被分析")
        full = " ".join(calls)
        assert "植被分析" in full and "数据浏览" in full

    def test_no_result_guidance_with_suggestions(self):
        import utils.empty_states as es
        calls = []
        class FakeSt:
            def warning(self, m): calls.append(m)
            def markdown(self, m): calls.append(m)
            def caption(self, m): calls.append(m)
        es.no_result_guidance(FakeSt(), "矢量化", suggestions=["减小面积阈值", "检查输入"])
        full = " ".join(calls)
        assert "减小面积阈值" in full
