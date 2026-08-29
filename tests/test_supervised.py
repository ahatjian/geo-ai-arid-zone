"""
监督分类训练 (supervised) 模块单元测试
测试: 特征构建 + 采样 + 训练 + 预测 + 精度评估
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def _make_synthetic_bands():
    """构造 3 类可分离的合成 6 波段数据 (植被/裸地/水体)"""
    np.random.seed(42)
    H, W = 60, 60
    # 类别标签 (0=植被, 1=裸地, 2=水体)
    labels = np.zeros((H, W), dtype=np.int16)
    labels[:, :20] = 0   # 左 1/3 植被
    labels[:, 20:40] = 1  # 中 1/3 裸地
    labels[:, 40:] = 2    # 右 1/3 水体

    bands = np.zeros((6, H, W), dtype=np.float32)
    # 植被: NIR 高, R 低
    bands[0, labels == 0] = 0.05   # B
    bands[1, labels == 0] = 0.10   # G
    bands[2, labels == 0] = 0.06   # R
    bands[3, labels == 0] = 0.45   # NIR
    bands[4, labels == 0] = 0.20   # SWIR1
    bands[5, labels == 0] = 0.15   # SWIR2
    # 裸地: 各波段中等偏亮
    bands[0, labels == 1] = 0.20
    bands[1, labels == 1] = 0.25
    bands[2, labels == 1] = 0.30
    bands[3, labels == 1] = 0.35
    bands[4, labels == 1] = 0.40
    bands[5, labels == 1] = 0.38
    # 水体: 各波段低, G > NIR
    bands[0, labels == 2] = 0.04
    bands[1, labels == 2] = 0.08
    bands[2, labels == 2] = 0.05
    bands[3, labels == 2] = 0.02
    bands[4, labels == 2] = 0.01
    bands[5, labels == 2] = 0.01

    bands_dict = {
        "B": bands[0], "G": bands[1], "R": bands[2],
        "NIR": bands[3], "SWIR1": bands[4], "SWIR2": bands[5],
    }
    return bands_dict, labels


class TestFeatureBuilding:
    def test_build_feature_stack(self):
        from utils.supervised import build_feature_stack
        bands_dict, _ = _make_synthetic_bands()
        stack = build_feature_stack(bands_dict, add_indices=True)
        assert stack.shape == (60, 60, 9)  # 6 波段 + 3 指数

        stack2 = build_feature_stack(bands_dict, add_indices=False)
        assert stack2.shape == (60, 60, 6)

    def test_build_feature_stack_missing_band(self):
        from utils.supervised import build_feature_stack
        bands_dict, _ = _make_synthetic_bands()
        del bands_dict["NIR"]
        with pytest.raises(ValueError):
            build_feature_stack(bands_dict)


class TestSampling:
    def test_sample_from_reference(self):
        from utils.supervised import sample_from_reference
        _, labels = _make_synthetic_bands()
        # 用 labels 本身作为参考分类图 (跳过 0 类)
        indices, sampled_labels = sample_from_reference(
            labels, n_samples_per_class=100, skip_zero=False,
        )
        assert indices.shape[0] == sampled_labels.shape[0]
        assert indices.shape[1] == 2
        # 3 类都有样本
        assert len(np.unique(sampled_labels)) == 3
        # 每类最多 100 个
        for cls in np.unique(sampled_labels):
            assert np.sum(sampled_labels == cls) <= 100

    def test_sample_from_reference_skip_zero(self):
        from utils.supervised import sample_from_reference
        _, labels = _make_synthetic_bands()
        indices, sampled = sample_from_reference(labels, skip_zero=True)
        # 跳过 0 类 (植被)
        assert 0 not in np.unique(sampled)


class TestTraining:
    def test_train_classifier_rf(self):
        from utils.supervised import train_classifier, build_feature_stack
        bands_dict, labels = _make_synthetic_bands()
        stack = build_feature_stack(bands_dict, add_indices=True)
        H, W, nf = stack.shape
        flat = stack.reshape(-1, nf)
        y = labels.reshape(-1)
        # 采样训练
        rng = np.random.RandomState(0)
        idx = rng.choice(H * W, 500, replace=False)
        model = train_classifier(flat[idx], y[idx], classifier="random_forest", n_estimators=50)
        assert model is not None
        # 预测精度应较高 (可分离数据)
        pred = model.predict(flat[idx])
        acc = np.mean(pred == y[idx])
        assert acc > 0.9

    def test_train_all_classifiers(self):
        from utils.supervised import train_classifier, build_feature_stack
        bands_dict, labels = _make_synthetic_bands()
        stack = build_feature_stack(bands_dict, add_indices=True)
        flat = stack.reshape(-1, stack.shape[2])
        y = labels.reshape(-1)
        rng = np.random.RandomState(0)
        idx = rng.choice(len(flat), 200, replace=False)
        for clf in ["random_forest", "svm", "knn", "mlp"]:
            model = train_classifier(flat[idx], y[idx], classifier=clf)
            assert model is not None

    def test_predict_image(self):
        from utils.supervised import train_classifier, predict_image, build_feature_stack
        bands_dict, labels = _make_synthetic_bands()
        stack = build_feature_stack(bands_dict, add_indices=True)
        flat = stack.reshape(-1, stack.shape[2])
        y = labels.reshape(-1)
        rng = np.random.RandomState(0)
        idx = rng.choice(len(flat), 500, replace=False)
        model = train_classifier(flat[idx], y[idx], classifier="random_forest", n_estimators=50)
        pred = predict_image(model, stack)
        assert pred.shape == (60, 60)
        assert pred.min() >= 0 and pred.max() <= 2


class TestEvaluation:
    def test_evaluate_classification(self):
        from utils.supervised import evaluate_classification
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        acc = evaluate_classification(y_true, y_pred, ["植被", "裸地", "水体"])
        assert "confusion_matrix" in acc
        assert "overall_accuracy" in acc
        assert "kappa" in acc
        assert "per_class_f1" in acc
        assert 0 <= acc["overall_accuracy"] <= 1

    def test_assess_supervised_classification(self):
        from utils.supervised import assess_supervised_classification
        bands_dict, labels = _make_synthetic_bands()
        indices, sampled_labels = __import__("utils.supervised", fromlist=["sample_from_reference"]).sample_from_reference(
            labels, n_samples_per_class=200, skip_zero=False,
        )
        result = assess_supervised_classification(
            bands_dict, indices, sampled_labels,
            classifier="random_forest", add_indices=True, test_ratio=0.3,
            n_estimators=50,
        )
        assert result.prediction is not None
        assert result.accuracy is not None
        assert result.prediction.shape == (60, 60)
        # 可分离数据应有较高精度
        assert result.summary["overall_accuracy"] > 0.9
