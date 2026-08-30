"""
结果持久化模块单元测试
=======================
覆盖: 保存(npy/csv/geojson/文件)、列出、加载、删除、清空、打包
"""
import os
import json
import zipfile

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clean_results(tmp_path, monkeypatch):
    """每个测试用临时目录, 测试后自动清理"""
    import utils.results_store as rs

    monkeypatch.setattr(rs, "RESULTS_DIR", str(tmp_path / "results"))
    yield
    # 清理
    if os.path.isdir(rs.RESULTS_DIR):
        import shutil
        shutil.rmtree(rs.RESULTS_DIR, ignore_errors=True)


# ============================================
# 测试
# ============================================

def test_save_result_npy():
    """保存 numpy 数组"""
    from utils import results_store as rs

    arr = np.arange(12).reshape(3, 4)
    r = rs.save_result("测试数组", arr, kind="npy", meta={"area": "塔里木"})

    assert r["name"] == "测试数组"
    assert r["kind"] == "npy"
    assert r["meta"]["area"] == "塔里木"
    assert os.path.exists(r["path"])
    assert r["size_bytes"] > 0


def test_save_result_csv():
    """保存 DataFrame 为 csv"""
    import pandas as pd
    from utils import results_store as rs

    df = pd.DataFrame({"类别": ["水体", "植被"], "面积": [1.5, 3.2]})
    r = rs.save_result("面积统计", df, kind="csv")

    assert r["kind"] == "csv"
    assert os.path.exists(r["path"])
    # 回读验证
    loaded = rs.load_result(r["id"])
    assert isinstance(loaded, pd.DataFrame)
    assert list(loaded.columns) == ["类别", "面积"]


def test_save_result_geojson():
    """保存 GeoJSON 文本"""
    from utils import results_store as rs

    geojson_str = '{"type":"FeatureCollection","features":[]}'
    r = rs.save_result("边界", geojson_str, kind="geojson")

    assert r["kind"] == "geojson"
    loaded = rs.load_result(r["id"])
    assert isinstance(loaded, str)
    assert json.loads(loaded)["type"] == "FeatureCollection"


def test_save_result_file():
    """保存任意二进制文件"""
    from utils import results_store as rs

    content = b"fake-tiff-binary-content"
    r = rs.save_result_file(content, "ndvi_map.tif")

    assert r["ext"] == "tif"
    assert r["name"] == "ndvi_map.tif"
    assert os.path.exists(r["path"])

    loaded = rs.load_result(r["id"])
    assert loaded == content


def test_list_results_sorted():
    """列出结果, 按时间降序"""
    import time
    from utils import results_store as rs

    r1 = rs.save_result("第一个", np.array([1]), kind="npy")
    time.sleep(0.01)  # 确保微秒时间戳可区分
    r2 = rs.save_result("第二个", np.array([2]), kind="npy")

    results = rs.list_results()
    ids = [r["id"] for r in results]
    # r2 更晚, 应排在前面
    assert ids[0] == r2["id"]
    assert ids[1] == r1["id"]
    assert len(results) == 2


def test_get_result():
    """按 id 获取元数据"""
    from utils import results_store as rs

    r = rs.save_result("目标", np.array([9]), kind="npy", meta={"k": "v"})
    got = rs.get_result(r["id"])
    assert got is not None
    assert got["name"] == "目标"
    assert got["meta"]["k"] == "v"


def test_delete_result():
    """删除单个结果"""
    from utils import results_store as rs

    r = rs.save_result("要删除", np.array([1]), kind="npy")
    assert rs.delete_result(r["id"]) is True
    assert rs.get_result(r["id"]) is None
    # 删除不存在的返回 False
    assert rs.delete_result("nonexistent") is False


def test_clear_results():
    """清空全部"""
    from utils import results_store as rs

    rs.save_result("a", np.array([1]), kind="npy")
    rs.save_result("b", np.array([2]), kind="npy")
    n = rs.clear_results()
    assert n == 2
    assert rs.list_results() == []


def test_package_results():
    """打包多个结果为 zip"""
    from utils import results_store as rs

    r1 = rs.save_result("数组A", np.array([1, 2, 3]), kind="npy")
    r2 = rs.save_result("表格B", [{"a": 1, "b": 2}], kind="csv")

    path, data = rs.package_results([r1["id"], r2["id"]])

    assert path.endswith(".zip")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        # 每个结果含数据文件 + meta.json
        assert any("数组A" in n and "meta.json" in n for n in names)
        assert any("表格B" in n and "data.csv" in n for n in names)


def test_package_results_missing_id():
    """打包包含不存在 id: 静默跳过"""
    from utils import results_store as rs

    r1 = rs.save_result("存在的", np.array([1]), kind="npy")
    path, data = rs.package_results([r1["id"], "nonexistent_id"])
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    # 只打包存在的
    assert len([n for n in names if "meta.json" in n]) == 1


def test_get_store_info():
    """结果库概要信息"""
    from utils import results_store as rs

    rs.save_result("x", np.zeros((10, 10)), kind="npy")
    info = rs.get_store_info()
    assert info["result_count"] == 1
    assert info["total_bytes"] > 0


def test_save_invalid_kind():
    """非法 kind 抛 ValueError"""
    from utils import results_store as rs

    with pytest.raises(ValueError):
        rs.save_result("x", np.array([1]), kind="unknown_kind")


def test_load_nonexistent():
    """加载不存在的 id 返回 None"""
    from utils import results_store as rs

    assert rs.load_result("no_such_id") is None
