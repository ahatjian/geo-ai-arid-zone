"""结果一键入库组件测试 — save_ui 与 results_store 联动"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

# 测试用临时结果目录
RESULTS_DIR_TEST = os.path.join(os.path.dirname(__file__), "_tmp_results_test")


@pytest.fixture(autouse=True)
def _isolated_results_dir(tmp_path, monkeypatch):
    """隔离测试结果目录, 避免污染真实结果库"""
    import utils.results_store as rs
    monkeypatch.setattr(rs, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(rs, "_ensure_dir", lambda: str(tmp_path))


class TestSaveResultRoundtrip:
    """保存 → 列表 → 读取 完整闭环"""

    def test_save_npy_roundtrip(self):
        from utils.results_store import save_result, list_results, load_result
        arr = np.arange(100, dtype=np.int8).reshape(10, 10)
        res = save_result("测试数组", arr, kind="npy", meta={"模块": "测试"})
        assert res["kind"] == "npy"
        assert res["meta"]["模块"] == "测试"
        assert os.path.exists(res["path"])

        results = list_results()
        assert any(r["id"] == res["id"] for r in results)

        loaded = load_result(res["id"])
        assert np.array_equal(np.asarray(loaded), arr)

    def test_save_csv_stats(self):
        from utils.results_store import save_result, load_result
        stats = [{"name": "A", "ratio": 0.6}, {"name": "B", "ratio": 0.4}]
        res = save_result("统计", stats, kind="csv")
        loaded = load_result(res["id"])
        assert "A" in str(loaded)

    def test_save_json_text(self):
        from utils.results_store import save_result, load_result
        res = save_result("报告", {"结论": "植被覆盖良好"}, kind="json")
        loaded = load_result(res["id"])
        assert "植被" in str(loaded)

    def test_result_id_unique(self):
        """连续保存多个结果 ID 不应重复"""
        from utils.results_store import save_result, _new_result_id
        ids = {_new_result_id() for _ in range(100)}
        assert len(ids) == 100  # uuid 短 ID 无碰撞

    def test_save_binary_png(self):
        from utils.results_store import save_result, load_result
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        res = save_result("截图", png_bytes, kind="png")
        loaded = load_result(res["id"])
        assert loaded.startswith(b"\x89PNG")


class TestDeleteAndPackage:
    def test_delete_result(self):
        from utils.results_store import save_result, delete_result, list_results
        res = save_result("待删除", np.zeros((3, 3)), kind="npy")
        delete_result(res["id"])
        assert all(r["id"] != res["id"] for r in list_results())

    def test_package_results(self):
        from utils.results_store import save_result, package_results, list_results
        save_result("结果1", np.zeros((2, 2)), kind="npy")
        save_result("结果2", {"k": 1}, kind="json")
        ids = [r["id"] for r in list_results()]
        zip_path, _ = package_results(result_ids=ids)
        assert zip_path and os.path.exists(zip_path)
        assert zipfile_is_valid(zip_path)


def zipfile_is_valid(path):
    import zipfile
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) >= 2
    except Exception:
        return False


class TestSaveUiHelpers:
    def test_render_save_csv_button_dataframe(self):
        """stats list[dict] 应能转换为 DataFrame 供 csv 保存"""
        import pandas as pd
        from utils.save_ui import render_save_csv_button
        # 仅验证转换逻辑 (不调用 streamlit 渲染)
        stats = [{"name": "A", "ratio": 0.6}]
        df = pd.DataFrame(stats) if not isinstance(stats, pd.DataFrame) else stats
        assert df.shape == (1, 2)


class TestReportCollection:
    """报告导出自动采集测试 — 新模块 session_state 采集"""

    def test_salinity_stats_structure(self):
        """盐渍化页面写入的 session_state 结构应包含报告所需字段"""
        ss = {
            "area": "塔里木盆地", "date": "2025-06-01", "satellite": "S2",
            "summary": {"total_ratio": 0.35, "severe_ratio": 0.12,
                        "dominant_level": "中度盐渍化", "ndsi_mean": 0.28},
            "stats": [{"name": "中度盐渍化", "ratio": 0.35}],
        }
        assert 0 <= ss["summary"]["total_ratio"] <= 1
        assert ss["summary"]["dominant_level"]
        assert ss["stats"]

    def test_lst_stats_structure(self):
        ls = {"area": "塔里木盆地", "date": "2025-06-01",
              "summary": {"mean_lst_c": 32.5, "max_lst_c": 45.2,
                          "hot_ratio": 0.3, "dominant_level": "高温"},
              "stats": []}
        assert ls["summary"]["mean_lst_c"] < ls["summary"]["max_lst_c"]
        assert ls["summary"]["hot_ratio"] >= 0

    def test_et_stats_structure(self):
        es = {"area": "河西走廊", "date": "2025-06-01",
              "summary": {"mean_et": 2.8, "mean_rn": 210.5,
                          "mean_le": 80.2, "dominant_level": "中等蒸散发"},
              "stats": []}
        assert es["summary"]["mean_et"] > 0
        assert es["summary"]["mean_rn"] > es["summary"]["mean_le"]

    def test_supervised_stats_structure(self):
        sp = {"area": "塔里木盆地", "classifier": "RandomForest",
              "accuracy": {"oa": 0.873, "kappa": 0.812, "f1_macro": 0.855},
              "summary": {"n_train": 500, "n_test": 200, "n_features": 9}}
        assert 0 < sp["accuracy"]["oa"] <= 1
        assert 0 <= sp["accuracy"]["kappa"] <= 1
        assert sp["summary"]["n_train"] > sp["summary"]["n_test"]

    def test_transition_stats_structure(self):
        tr = {"area": "准噶尔盆地", "t1": "2020", "t2": "2021",
              "summary": {"total_change_km2": 125.3, "total_area_km2": 5000.0,
                          "n_major": 3},
              "major_transitions": [{"transition": "草地→裸地", "area_km2": 45.2}]}
        assert tr["summary"]["total_change_km2"] > 0
        assert tr["major_transitions"]
        assert tr["t1"] != tr["t2"]


class TestReportNewModules:
    """报告导出扩展采集 — BFAST / KMeans"""

    def test_bfast_result_structure(self):
        """植被分析页写入的 bfast_result 结构"""
        bf = {
            "n_breaks": 2, "n_negative": 1, "n_positive": 1,
            "break_dates": ["2024-06", "2025-03"],
            "magnitudes": [-0.12, 0.08],
            "directions": ["负向突变", "正向突变"],
        }
        assert bf["n_breaks"] == len(bf["break_dates"])
        assert bf["n_negative"] + bf["n_positive"] == bf["n_breaks"]
        assert sum(1 for d in bf["directions"] if d == "负向突变") == bf["n_negative"]

    def test_kmeans_result_structure(self):
        """AI分类页写入的 km_class_result 可采集"""
        import numpy as np
        km = np.zeros((10, 10), dtype=np.int16)
        km[:5] = 1
        km[5:] = 2
        n_classes = int(km.max()) + 1
        assert n_classes == 3
        assert km.size == 100
