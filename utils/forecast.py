"""
干旱趋势预测模块
================
支持三种预测方法:
  - SARIMA (statsmodels) — 统计预测，带置信区间
  - LSTM (PyTorch)      — 深度学习，捕捉非线性模式
  - Holt-Winters (statsmodels) — 指数平滑，轻量级备选

核心流程: NDVI时序 → 训练/测试分割 → 模型拟合 → 多步预测 → 评估

参考:
  - 慧天平台 FYDI 30-90天预测
  - 西北干旱监测预测业务服务综合系统
  - Box-Jenkins SARIMA 方法论

依赖: numpy, pandas, scipy, sklearn, statsmodels (必选)
      torch (可选, LSTM 预测需要)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List, Tuple, Union, Literal
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ---- 条件导入 ----
_IMPORTS = {"statsmodels": False, "torch": False}

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _IMPORTS["statsmodels"] = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _IMPORTS["torch"] = True
except ImportError:
    pass

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats as scipy_stats

# ---- 条件缓存 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=True)
    else:
        def noop(func):
            return func
        return noop


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ForecastResult:
    """预测结果容器"""
    method: str                          # 预测方法名
    forecast_values: np.ndarray          # 预测值 (forecast_steps,)
    forecast_steps: int                  # 预测步数
    confidence_lower: Optional[np.ndarray] = None  # 下置信区间
    confidence_upper: Optional[np.ndarray] = None  # 上置信区间
    metrics: Dict[str, float] = field(default_factory=dict)  # 评估指标
    model_params: Dict = field(default_factory=dict)  # 模型参数
    historical_values: Optional[np.ndarray] = None  # 历史值
    historical_dates: Optional[List[str]] = None    # 历史日期
    forecast_dates: Optional[List[str]] = None      # 预测日期


# ============================================================
# 工具函数
# ============================================================

def prepare_ndvi_timeseries(
    ndvi_stack: np.ndarray,
    method: str = "mean",
    dates: Optional[List[str]] = None,
    mask_nodata: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 NDVI 时序立方体 (H, W, T) 提取区域均值时序

    参数:
        ndvi_stack: (H, W, T) 三维数组
        method: "mean" | "median" | "pixel" 聚合方法
        dates: 日期标签列表 (可选)
        mask_nodata: 是否屏蔽无效值

    返回:
        (values, times) — values: (T,) 时序值, times: (T,) 时间索引
    """
    if ndvi_stack.ndim != 3:
        raise ValueError(f"ndvi_stack 必须是 3D 数组 (H,W,T), 实际: {ndvi_stack.ndim}D")

    H, W, T = ndvi_stack.shape
    values = np.zeros(T, dtype=np.float64)

    for t in range(T):
        frame = ndvi_stack[:, :, t].astype(np.float64)
        if mask_nodata:
            frame = frame[np.isfinite(frame) & (frame > -0.5) & (frame < 1.5)]

        if len(frame) == 0:
            values[t] = np.nan
        elif method == "median":
            values[t] = np.nanmedian(frame)
        else:
            values[t] = np.nanmean(frame)

    times = np.arange(T, dtype=np.float64)

    # 简单线性插值填充 NaN
    nan_mask = np.isnan(values)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) >= 2:
            values = np.interp(times, valid_idx, values[valid_idx])

    return values, times


def split_train_test(
    values: np.ndarray,
    test_ratio: float = 0.2,
    min_train: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    划分训练/测试集

    参数:
        values: 一维时序
        test_ratio: 测试比例 (0~1)
        min_train: 最少训练样本数

    返回:
        (train_values, test_values)
    """
    n = len(values)
    split_point = max(min_train, int(n * (1 - test_ratio)))
    return values[:split_point].copy(), values[split_point:].copy()


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    评估预测精度

    返回:
        dict: {"RMSE": ..., "MAE": ..., "MAPE": ..., "R2": ..., "N": ...}
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) < 2:
        return {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0, "R2": 0.0, "N": len(actual)}

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # MAPE (避免除零)
    mape_mask = np.abs(actual) > 1e-6
    if mape_mask.any():
        mape = np.mean(np.abs((actual[mape_mask] - predicted[mape_mask]) / actual[mape_mask])) * 100
    else:
        mape = 0.0

    r2 = r2_score(actual, predicted) if len(actual) > 2 else 0.0

    return {"RMSE": round(rmse, 6), "MAE": round(mae, 6),
            "MAPE": round(mape, 2), "R2": round(r2, 4), "N": len(actual)}


# ============================================================
# 方法 1: SARIMA 预测
# ============================================================

def forecast_sarima(
    values: np.ndarray,
    forecast_steps: int = 12,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    confidence_level: float = 0.80,
    auto_order: bool = True,
) -> ForecastResult:
    """
    SARIMA 时序预测 (带自动定阶)

    参数:
        values: 一维时序数组
        forecast_steps: 预测步数 (如 12=预测未来12个时相)
        order: (p, d, q) ARIMA 阶数
        seasonal_order: (P, D, Q, s) 季节性阶数, None=自动
        confidence_level: 置信水平 (0~1)
        auto_order: 是否尝试自动选择 order

    返回:
        ForecastResult
    """
    if not _IMPORTS["statsmodels"]:
        raise ImportError("SARIMA 预测需要 statsmodels: pip install statsmodels")

    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < 6:
        raise ValueError(f"SARIMA 至少需要 6 个样本, 实际: {n}")

    if seasonal_order is None:
        # 自动推断季节性 — 如果 T>=12 用 12 步周期
        if n >= 12:
            seasonal_order = (0, 1, 1, min(12, n // 2))
        else:
            seasonal_order = (0, 0, 0, 0)

    if auto_order and n >= 10:
        order = _auto_arima_order(values, seasonal_order[3])

    try:
        model = SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

        forecast_result = fitted.get_forecast(steps=forecast_steps)
        forecast_values = forecast_result.predicted_mean
        ci = forecast_result.conf_int(alpha=1 - confidence_level)

        return ForecastResult(
            method="SARIMA",
            forecast_values=forecast_values,
            forecast_steps=forecast_steps,
            confidence_lower=ci[:, 0],
            confidence_upper=ci[:, 1],
            model_params={
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": round(fitted.aic, 2),
                "bic": round(fitted.bic, 2),
            },
            historical_values=values,
        )
    except Exception as e:
        # 降级到简单 ARIMA
        try:
            model = SARIMAX(values, order=(1, 0, 0), enforce_stationarity=False)
            fitted = model.fit(disp=False, maxiter=100)
            forecast = fitted.get_forecast(steps=forecast_steps).predicted_mean
            return ForecastResult(
                method="SARIMA(fallback)",
                forecast_values=forecast,
                forecast_steps=forecast_steps,
                model_params={"order": (1, 0, 0), "fallback_reason": str(e)[:100]},
                historical_values=values,
            )
        except Exception:
            raise RuntimeError(f"SARIMA 预测失败: {e}")


def _auto_arima_order(
    values: np.ndarray,
    seasonal_period: int = 0,
    max_p: int = 3,
    max_q: int = 3,
) -> Tuple[int, int, int]:
    """简易自动 ARIMA 定阶 (基于 AIC)"""
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                if seasonal_period > 0:
                    model = SARIMAX(values, order=(p, 1, q),
                                    seasonal_order=(0, 1, 1, seasonal_period),
                                    enforce_stationarity=False)
                else:
                    model = SARIMAX(values, order=(p, 1, q),
                                    enforce_stationarity=False)
                fitted = model.fit(disp=False, maxiter=100)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, 1, q)
            except Exception:
                continue

    return best_order


# ============================================================
# 方法 2: LSTM 预测 (PyTorch)
# ============================================================

class _LSTMForecaster(nn.Module):
    """轻量级 LSTM 预测器"""

    def __init__(self, input_size: int = 1, hidden_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _create_lstm_sequences(
    values: np.ndarray,
    lookback: int,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """创建 LSTM 训练序列"""
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values.reshape(-1, 1))

    scaled = scaler.transform(values.reshape(-1, 1)).flatten()
    X, y = [], []
    for i in range(len(scaled) - lookback):
        X.append(scaled[i:i + lookback])
        y.append(scaled[i + lookback])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X_tensor, y_tensor, scaler


def forecast_lstm(
    values: np.ndarray,
    forecast_steps: int = 12,
    lookback: int = 6,
    hidden_size: int = 32,
    num_layers: int = 2,
    epochs: int = 200,
    learning_rate: float = 0.001,
    patience: int = 20,
    verbose: bool = False,
) -> ForecastResult:
    """
    LSTM 时序预测 (PyTorch)

    参数:
        values: 一维时序数组
        forecast_steps: 预测步数
        lookback: 回看窗口大小
        hidden_size: 隐藏层大小
        num_layers: LSTM 层数
        epochs: 训练轮数
        learning_rate: 学习率
        patience: 早停耐心值
        verbose: 是否打印训练信息

    返回:
        ForecastResult
    """
    if not _IMPORTS["torch"]:
        raise ImportError("LSTM 预测需要 PyTorch: pip install torch")

    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < lookback + 3:
        raise ValueError(f"LSTM 至少需要 lookback+3={lookback+3} 个样本, 实际: {n}")

    # 准备序列
    X, y, scaler = _create_lstm_sequences(values, lookback)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _LSTMForecaster(
        input_size=1, hidden_size=hidden_size,
        num_layers=num_layers, dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练 + 早停
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  早停 @ epoch {epoch+1}, loss={best_loss:.6f}")
            break

    # 加载最佳模型
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # 多步预测 (迭代)
    last_seq = scaler.transform(values[-lookback:].reshape(-1, 1)).flatten()
    predictions_scaled = []

    with torch.no_grad():
        seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        for _ in range(forecast_steps):
            pred = model(seq).cpu().numpy()[0, 0]
            predictions_scaled.append(pred)
            # 滑动窗口
            seq = torch.cat([
                seq[:, 1:, :],
                torch.tensor([[[pred]]], dtype=torch.float32).to(device)
            ], dim=1)

    forecast_values = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()

    # 使用 Bootstrap 估计置信区间
    ci_lower, ci_upper = _bootstrap_confidence(
        forecast_values, values, n_bootstrap=100, ci=0.80
    )

    return ForecastResult(
        method="LSTM",
        forecast_values=forecast_values,
        forecast_steps=forecast_steps,
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        model_params={
            "lookback": lookback,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "final_loss": round(best_loss, 6),
        },
        historical_values=values,
    )


def _bootstrap_confidence(
    forecast: np.ndarray,
    history: np.ndarray,
    n_bootstrap: int = 100,
    ci: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap 法估计预测置信区间"""
    residuals = []
    # 基于历史序列的自相关残差
    for i in range(1, len(history)):
        residuals.append(history[i] - history[i - 1])

    residuals = np.array(residuals)
    if len(residuals) < 10:
        std = np.std(history) * 0.2
        z = scipy_stats.norm.ppf(0.5 + ci / 2)
        return forecast - z * std, forecast + z * std

    boot_samples = np.random.choice(residuals, size=(n_bootstrap, len(forecast)))
    cumsum_samples = np.cumsum(boot_samples, axis=1)
    forecast_broad = forecast.reshape(1, -1)
    simulated = forecast_broad + cumsum_samples

    alpha = (1 - ci) / 2
    lower = np.percentile(simulated, alpha * 100, axis=0)
    upper = np.percentile(simulated, (1 - alpha) * 100, axis=0)
    return lower, upper


# ============================================================
# 方法 3: Holt-Winters 指数平滑
# ============================================================

def forecast_holt_winters(
    values: np.ndarray,
    forecast_steps: int = 12,
    seasonal_periods: Optional[int] = None,
    trend: str = "add",
    seasonal: Optional[str] = None,
) -> ForecastResult:
    """
    Holt-Winters 指数平滑预测

    参数:
        values: 一维时序
        forecast_steps: 预测步数
        seasonal_periods: 季节周期, None=自动
        trend: "add" | "mul" | None
        seasonal: "add" | "mul" | None

    返回:
        ForecastResult
    """
    if not _IMPORTS["statsmodels"]:
        raise ImportError("Holt-Winters 预测需要 statsmodels: pip install statsmodels")

    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < 4:
        raise ValueError(f"Holt-Winters 至少需要 4 个样本, 实际: {n}")

    if seasonal_periods is None:
        seasonal_periods = min(12, max(2, n // 2))

    if seasonal is None:
        seasonal = "add" if seasonal_periods <= n else None

    try:
        if seasonal and seasonal_periods >= 2:
            model = ExponentialSmoothing(
                values,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
        else:
            model = ExponentialSmoothing(values, trend=trend)

        fitted = model.fit()

        forecast_values = fitted.forecast(forecast_steps)

        # 残差法估计置信区间
        residuals = values - fitted.fittedvalues
        valid_res = residuals[~np.isnan(residuals)]
        std_res = np.std(valid_res) if len(valid_res) > 0 else np.std(values) * 0.1
        z = scipy_stats.norm.ppf(0.90)  # 80% CI
        ci_half = z * std_res * np.sqrt(np.arange(1, forecast_steps + 1))

        return ForecastResult(
            method="Holt-Winters",
            forecast_values=forecast_values,
            forecast_steps=forecast_steps,
            confidence_lower=forecast_values - ci_half,
            confidence_upper=forecast_values + ci_half,
            model_params={
                "trend": trend,
                "seasonal": seasonal,
                "seasonal_periods": seasonal_periods,
            },
            historical_values=values,
        )
    except Exception as e:
        # 降级到简单指数平滑
        try:
            model = ExponentialSmoothing(values, trend=None)
            fitted = model.fit()
            forecast = fitted.forecast(forecast_steps)
            return ForecastResult(
                method="Holt-Winters(fallback)",
                forecast_values=forecast,
                forecast_steps=forecast_steps,
                model_params={"trend": None, "fallback_reason": str(e)[:100]},
                historical_values=values,
            )
        except Exception:
            raise RuntimeError(f"Holt-Winters 预测失败: {e}")


# ============================================================
# 统一预测接口
# ============================================================

def forecast_drought_trend(
    ndvi_stack: np.ndarray,
    forecast_steps: int = 12,
    method: Literal["sarima", "lstm", "holt_winters", "all"] = "sarima",
    test_ratio: float = 0.0,  # 0=使用全部数据进行预测
    dates: Optional[List[str]] = None,
    **kwargs,
) -> Union[ForecastResult, Dict[str, ForecastResult]]:
    """
    干旱趋势预测 — 统一入口

    参数:
        ndvi_stack: (H, W, T) NDVI 时序立方体
        forecast_steps: 预测未来步数
        method: 预测方法
        test_ratio: 测试比例 (>0 时评估预测精度)
        dates: 日期标签
        **kwargs: 传递给具体预测方法的参数

    返回:
        ForecastResult 或 Dict[str, ForecastResult]
    """
    # 提取均值时序
    values, _ = prepare_ndvi_timeseries(ndvi_stack)

    # 如果有测试集，评估精度
    metrics = {}
    if test_ratio > 0:
        train_values, test_values = split_train_test(values, test_ratio=test_ratio)
        use_values = train_values
    else:
        use_values = values

    if method == "all":
        results = {}
        for m in ["sarima", "lstm", "holt_winters"]:
            try:
                result = _forecast_single(use_values, m, forecast_steps, **kwargs)
                if test_ratio > 0:
                    # 评估: 预测训练集末尾的 test_len 个点
                    test_len = len(values) - len(use_values)
                    _, test_actual = split_train_test(values, test_ratio=test_ratio)
                    test_pred = _backtest(use_values, m, test_len, **kwargs)
                    result.metrics = evaluate_forecast(test_actual, test_pred)
                results[m] = result
            except Exception as e:
                results[m] = ForecastResult(
                    method=m, forecast_values=np.array([]),
                    forecast_steps=forecast_steps,
                    model_params={"error": str(e)},
                )
        return results
    else:
        result = _forecast_single(use_values, method, forecast_steps, **kwargs)
        if test_ratio > 0:
            test_len = len(values) - len(use_values)
            _, test_actual = split_train_test(values, test_ratio=test_ratio)
            test_pred = _backtest(use_values, method, test_len, **kwargs)
            result.metrics = evaluate_forecast(test_actual, test_pred)
        return result


def _forecast_single(
    values: np.ndarray,
    method: str,
    forecast_steps: int,
    **kwargs,
) -> ForecastResult:
    """单方法预测调度"""
    if method == "sarima":
        return forecast_sarima(
            values, forecast_steps=forecast_steps,
            order=kwargs.get("order", (1, 1, 1)),
            seasonal_order=kwargs.get("seasonal_order", None),
            auto_order=kwargs.get("auto_order", True),
        )
    elif method == "lstm":
        return forecast_lstm(
            values, forecast_steps=forecast_steps,
            lookback=kwargs.get("lookback", 6),
            hidden_size=kwargs.get("hidden_size", 32),
            num_layers=kwargs.get("num_layers", 2),
            epochs=kwargs.get("epochs", 200),
            patience=kwargs.get("patience", 20),
            verbose=kwargs.get("verbose", False),
        )
    elif method == "holt_winters":
        return forecast_holt_winters(
            values, forecast_steps=forecast_steps,
            seasonal_periods=kwargs.get("seasonal_periods", None),
            trend=kwargs.get("trend", "add"),
            seasonal=kwargs.get("seasonal", None),
        )
    else:
        raise ValueError(f"未知预测方法: {method}, 可选: sarima, lstm, holt_winters, all")


def _backtest(
    values: np.ndarray,
    method: str,
    test_len: int,
    **kwargs,
) -> np.ndarray:
    """回测: 用前 N-test_len 个点预测后 test_len 个点"""
    train = values[:-test_len]
    result = _forecast_single(train, method, test_len, **kwargs)
    return result.forecast_values


# ============================================================
# 预测可视化 (matplotlib)
# ============================================================

def plot_forecast(
    result: ForecastResult,
    title: str = "干旱趋势预测",
    figsize: Tuple[int, int] = (12, 6),
    show_confidence: bool = True,
    return_fig: bool = False,
):
    """
    绘制预测结果图

    参数:
        result: ForecastResult 预测结果
        title: 图表标题
        figsize: 图表大小
        show_confidence: 是否显示置信区间
        return_fig: 是否返回 figure (用于 Streamlit)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=figsize)

    n_hist = len(result.historical_values) if result.historical_values is not None else 0
    n_future = result.forecast_steps

    total_steps = n_hist + n_future
    hist_x = np.arange(n_hist)

    # 历史曲线
    if n_hist > 0:
        ax.plot(hist_x, result.historical_values, "o-",
                color="#2ecc71", linewidth=2, markersize=5,
                label="历史 NDVI", zorder=3)

    # 预测曲线
    future_x = np.arange(n_hist, total_steps)
    ax.plot(future_x, result.forecast_values, "s--",
            color="#e74c3c", linewidth=2, markersize=6,
            label=f"预测 ({result.method})", zorder=4)

    # 置信区间
    if show_confidence and result.confidence_lower is not None:
        ax.fill_between(
            future_x,
            result.confidence_lower,
            result.confidence_upper,
            alpha=0.2, color="#e74c3c",
            label=f"80% 置信区间",
        )

    # 分割线
    if n_hist > 0:
        ax.axvline(x=n_hist - 1, color="#999", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(n_hist - 1.5, ax.get_ylim()[1] * 0.95, "历史 | 预测",
                ha="center", fontsize=9, color="#666")

    ax.set_xlabel("时相序号", fontsize=11)
    ax.set_ylabel("NDVI 均值", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # 指标注释
    if result.metrics:
        metrics_text = "  |  ".join(
            f"{k}: {v}" for k, v in result.metrics.items() if k != "N"
        )
        ax.text(0.02, 0.02, f"评估: {metrics_text}",
                transform=ax.transAxes, fontsize=8, color="#666",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()


def plot_forecast_comparison(
    results: Dict[str, ForecastResult],
    title: str = "多方法预测对比",
    figsize: Tuple[int, int] = (12, 6),
    return_fig: bool = False,
):
    """
    多方法预测对比图

    参数:
        results: {"SARIMA": ForecastResult, "LSTM": ForecastResult, ...}
        title: 图表标题
        figsize: 图表大小
        return_fig: 是否返回 figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=figsize)

    colors = {"SARIMA": "#3498db", "LSTM": "#e74c3c",
              "Holt-Winters": "#f39c12", "SARIMA(fallback)": "#95a5a6",
              "Holt-Winters(fallback)": "#bdc3c7"}

    # 历史数据 (取第一个有效结果)
    hist_values = None
    for r in results.values():
        if r.historical_values is not None:
            hist_values = r.historical_values
            break

    n_hist = len(hist_values) if hist_values is not None else 0
    if n_hist > 0:
        ax.plot(np.arange(n_hist), hist_values, "ko-",
                linewidth=2, markersize=5, label="历史 NDVI", zorder=5)

    # 各方法预测
    for method, result in results.items():
        if len(result.forecast_values) == 0:
            continue
        future_x = np.arange(n_hist, n_hist + result.forecast_steps)
        color = colors.get(method, "#999")
        ax.plot(future_x, result.forecast_values, "s--",
                color=color, linewidth=2, markersize=5,
                label=f"{method} (MAE={result.metrics.get('MAE', 'N/A')})",
                zorder=4)

    if n_hist > 0:
        ax.axvline(x=n_hist - 1, color="#999", linestyle="--", alpha=0.6)

    ax.set_xlabel("时相序号", fontsize=11)
    ax.set_ylabel("NDVI 均值", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()


# ============================================================
# 便捷: 从研究区数据一键预测
# ============================================================

@_cache(ttl=600)
def analyze_drought_forecast(
    ndvi_stack: np.ndarray,
    forecast_steps: int = 12,
    method: str = "all",
    test_ratio: float = 0.0,
    dates: Optional[List[str]] = None,
) -> Dict:
    """
    一键干旱预测分析 (缓存版)

    返回:
        dict: {
            "forecast_results": ForecastResult 或 dict,
            "summary": {"trend_direction": "改善/退化/稳定", "slope": float, ...},
            "values": ndarray,
            "dates": list,
        }
    """
    from utils.trend import theil_sen_slope

    values, _ = prepare_ndvi_timeseries(ndvi_stack)

    # 趋势判断
    try:
        slope = theil_sen_slope(values)
        if slope > 0.005:
            trend_dir = "🌿 改善"
        elif slope < -0.005:
            trend_dir = "🏜️ 退化"
        else:
            trend_dir = "➡️ 稳定"
    except Exception:
        slope = 0.0
        trend_dir = "未知"

    # 预测
    try:
        forecast = forecast_drought_trend(
            ndvi_stack,
            forecast_steps=forecast_steps,
            method=method,
            test_ratio=test_ratio,
            dates=dates,
        )
    except Exception as e:
        forecast = {"error": str(e)}

    return {
        "forecast_results": forecast,
        "summary": {
            "trend_direction": trend_dir,
            "sen_slope": round(slope, 6),
            "mean_ndvi": round(np.nanmean(values), 4),
            "std_ndvi": round(np.nanstd(values), 4),
            "n_timesteps": len(values),
            "forecast_steps": forecast_steps,
        },
        "values": values,
        "dates": dates,
    }
