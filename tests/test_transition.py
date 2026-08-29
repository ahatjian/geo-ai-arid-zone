"""
土地覆盖转移矩阵 (transition) 模块单元测试
测试: 转移矩阵计算 + 净变化 + 主要转移方向
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


@pytest.fixture
def mock_transition_data():
    """模拟双时相分类图 (6类)，含明显转移"""
    np.random.seed(42)
    H, W = 60, 60
    class_t1 = np.random.randint(0, 6, (H, W)).astype(np.int16)
    # T2: 部分像元发生转移 (类别 2 → 3, 4 → 1)
    class_t2 = class_t1.copy()
    mask = class_t1 == 2
    class_t2[mask] = np.where(np.random.rand(H, W)[mask] > 0.5, 3, 2)
    mask2 = class_t1 == 4
    class_t2[mask2] = np.where(np.random.rand(H, W)[mask2] > 0.5, 1, 4)
    return class_t1, class_t2


class TestTransitionMatrix:
    def test_compute_transition_matrix(self, mock_transition_data):
        from utils.transition import compute_transition_matrix
        class_t1, class_t2 = mock_transition_data
        area_matrix, pixel_matrix = compute_transition_matrix(
            class_t1, class_t2, n_classes=6, pixel_size_m=10
        )
        assert area_matrix.shape == (6, 6)
        assert pixel_matrix.shape == (6, 6)
        # 总像元数守恒
        assert np.sum(pixel_matrix) == 60 * 60
        # 对角线为未变化像元
        assert np.trace(pixel_matrix) >= 0

    def test_net_change(self, mock_transition_data):
        from utils.transition import compute_transition_matrix, net_change
        class_t1, class_t2 = mock_transition_data
        area_matrix, _ = compute_transition_matrix(class_t1, class_t2, n_classes=6, pixel_size_m=10)
        names = [f"类型{i}" for i in range(6)]
        nc = net_change(area_matrix, names)
        assert len(nc) == 6
        # 净变化之和应近似为 0 (面积守恒)
        net_sum = sum(x["net_km2"] for x in nc)
        assert abs(net_sum) < 0.01

    def test_find_major_transitions(self, mock_transition_data):
        from utils.transition import compute_transition_matrix, find_major_transitions
        class_t1, class_t2 = mock_transition_data
        area_matrix, _ = compute_transition_matrix(class_t1, class_t2, n_classes=6, pixel_size_m=10)
        names = [f"类型{i}" for i in range(6)]
        mt = find_major_transitions(area_matrix, names, top_n=5)
        assert len(mt) <= 5
        if mt:
            # 按面积降序
            areas = [x["area_km2"] for x in mt]
            assert areas == sorted(areas, reverse=True)

    def test_analyze_transition(self, mock_transition_data):
        from utils.transition import analyze_transition
        class_t1, class_t2 = mock_transition_data
        names = [f"类型{i}" for i in range(6)]
        result = analyze_transition(class_t1, class_t2, names, pixel_size_m=10)
        assert "area_matrix" in result
        assert "net_change" in result
        assert "major_transitions" in result
        assert "total_change_km2" in result

    def test_size_mismatch_raises(self):
        from utils.transition import compute_transition_matrix
        a = np.zeros((10, 10), dtype=np.int16)
        b = np.zeros((12, 12), dtype=np.int16)
        with pytest.raises(ValueError):
            compute_transition_matrix(a, b, n_classes=6)
