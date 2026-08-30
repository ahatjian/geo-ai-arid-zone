"""
自定义光谱指数计算器 (Band Math) 模块
========================================
提供灵活的波段运算能力，支持:
  - 12 种预设遥感指数 (NDVI/EVI/SAVI/NDWI/MNDWI/NDBI/BSI...)
  - 自定义波段运算表达式 (支持 + - * / 及 numpy 数学函数)

波段名约定 (Sentinel-2 / Landsat 统一):
  B, G, R, NIR, SWIR1, SWIR2 (对应 [B02/B2, B03/B3, B04/B4, B08/B5, B11/B6, B12/B7])

依赖: numpy
"""

import numpy as np
import ast
import operator
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ============================================================
# 波段名定义
# ============================================================

BAND_NAMES = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]

# 允许在表达式中使用的 numpy 函数 (安全白名单)
ALLOWED_NUMPY_FUNCS = [
    "sqrt", "log", "log10", "exp", "abs", "clip",
    "where", "maximum", "minimum", "power", "square",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "floor", "ceil", "round", "sign", "nan_to_num",
]

# ============================================================
# 预设指数库
# ============================================================

PRESET_INDICES = [
    {
        "name": "NDVI", "formula": "(NIR - R) / (NIR + R)",
        "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化植被指数 — 最常用的植被指标",
    },
    {
        "name": "EVI", "formula": "2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)",
        "cmap": "YlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "增强植被指数 — 避免高植被区饱和",
    },
    {
        "name": "SAVI", "formula": "(NIR - R) / (NIR + R + 0.5) * 1.5",
        "cmap": "YlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "土壤调节植被指数 — 降低土壤背景影响",
    },
    {
        "name": "GNDVI", "formula": "(NIR - G) / (NIR + G)",
        "cmap": "YlGn", "vmin": -1.0, "vmax": 1.0,
        "desc": "绿度归一化植被指数 — 对叶绿素更敏感",
    },
    {
        "name": "RVI", "formula": "NIR / R",
        "cmap": "Greens", "vmin": 0.0, "vmax": 8.0,
        "desc": "比值植被指数 — NIR 与红波段比值",
    },
    {
        "name": "DVI", "formula": "NIR - R",
        "cmap": "Greens", "vmin": -0.5, "vmax": 0.8,
        "desc": "差值植被指数 — NIR 与红波段差值",
    },
    {
        "name": "NDWI", "formula": "(G - NIR) / (G + NIR)",
        "cmap": "Blues", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化水体指数 — 水体提取",
    },
    {
        "name": "MNDWI", "formula": "(G - SWIR1) / (G + SWIR1)",
        "cmap": "Blues", "vmin": -1.0, "vmax": 1.0,
        "desc": "修正归一化水体指数 — 抑制建筑干扰",
    },
    {
        "name": "NDBI", "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "cmap": "Oranges", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化建筑指数 — 建设用地提取",
    },
    {
        "name": "BSI", "formula": "(SWIR1 + R - NIR - B) / (SWIR1 + R + NIR + B)",
        "cmap": "YlOrBr", "vmin": -1.0, "vmax": 1.0,
        "desc": "裸土指数 — 裸土/荒漠识别",
    },
    {
        "name": "NDMI", "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "cmap": "RdYlBu", "vmin": -1.0, "vmax": 1.0,
        "desc": "归一化水分指数 — 植被/土壤含水量",
    },
    {
        "name": "GRVI", "formula": "NIR / G",
        "cmap": "Greens", "vmin": 0.0, "vmax": 6.0,
        "desc": "绿红植被指数 — 绿度比值",
    },
]

PRESET_INDEX_NAMES = [p["name"] for p in PRESET_INDICES]


# ============================================================
# 波段运算求值器 (AST 白名单, 无 eval)
# ============================================================

# 二元运算映射 (仅允许算术运算符)
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# 一元运算映射
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 比较运算映射 (允许阈值比较, 配合 where 使用)
_CMP_OPS = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class _BandMathEvaluator(ast.NodeVisitor):
    """AST 白名单求值器: 仅允许常量/波段名/算术运算/白名单函数/比较。"""

    def __init__(self, namespace: Dict, allowed_funcs: List[str]):
        self._ns = namespace
        self._allowed = allowed_funcs
        self._max_nodes = 500  # 防表达式过深 (DoS)

    def generic_visit(self, node):
        raise ValueError(f"不允许的表达式元素: {type(node).__name__}")

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不允许的常量类型: {type(node.value).__name__}")

    def visit_Name(self, node):
        if node.id not in self._ns:
            raise ValueError(f"未知标识符: {node.id} (允许: B/G/R/NIR/SWIR1/SWIR2 + 数学函数)")
        return self._ns[node.id]

    def visit_BinOp(self, node):
        if type(node.op) not in _BIN_OPS:
            raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    def visit_UnaryOp(self, node):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"不允许的一元运算符: {type(node.op).__name__}")
        return _UNARY_OPS[type(node.op)](self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅允许调用白名单数学函数")
        if node.func.id not in self._allowed:
            raise ValueError(f"不允许的函数: {node.func.id}")
        func = getattr(np, node.func.id)
        args = [self.visit(a) for a in node.args]
        return func(*args)

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("仅支持单个比较")
        if type(node.ops[0]) not in _CMP_OPS:
            raise ValueError("不允许的比较运算符")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return _CMP_OPS[type(node.ops[0])](left, right)

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = np.logical_and(result, v)
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = np.logical_or(result, v)
            return result
        raise ValueError("不允许的逻辑运算符")

    def visit_Subscript(self, node):
        raise ValueError("不允许下标访问")

    def visit_Attribute(self, node):
        raise ValueError("不允许属性访问")


def _safe_eval_band_math(expression: str, namespace: Dict) -> np.ndarray:
    """AST 白名单求值, 替代 eval()。"""
    if len(expression) > 2000:
        raise ValueError("表达式过长")
    tree = ast.parse(expression, mode="eval")
    evaluator = _BandMathEvaluator(namespace, ALLOWED_NUMPY_FUNCS)
    return evaluator.visit(tree)


def get_band_arrays(bands_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将 (B, H, W) 波段数组转换为 {波段名: 数组} 字典

    参数:
        bands_data: (6, H, W) 多波段数据, 顺序 [B, G, R, NIR, SWIR1, SWIR2]

    返回:
        dict: {"B": ..., "G": ..., "R": ..., "NIR": ..., "SWIR1": ..., "SWIR2": ...}
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    if bands_data.shape[0] < 6:
        raise ValueError(f"至少需要 6 个波段, 实际: {bands_data.shape[0]}")

    # 自动缩放反射率 (DN → 0-1)
    if np.nanmedian(bands_data) > 10:
        bands_data = bands_data / 10000.0

    return {
        "B": bands_data[0],
        "G": bands_data[1],
        "R": bands_data[2],
        "NIR": bands_data[3],
        "SWIR1": bands_data[4],
        "SWIR2": bands_data[5],
    }


def evaluate_band_math(
    expression: str,
    bands: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    安全求值波段运算表达式 (AST 白名单, 不使用 eval)

    参数:
        expression: 波段运算表达式, 如 "(NIR - R) / (NIR + R)"
                    可使用波段名 (B/G/R/NIR/SWIR1/SWIR2) 和 numpy 数学函数
        bands: {波段名: 数组} 字典

    返回:
        result: 计算结果数组 (H, W)

    ⚠️ 安全说明: 通过 ast.parse + 白名单 NodeVisitor 求值，
       仅允许常量/波段名/算术运算符/白名单 numpy 函数，
       禁止属性访问/下标/调用外部函数，杜绝 RCE。
    """
    # 构建命名空间: 波段名 + numpy 白名单函数
    namespace = {name: arr for name, arr in bands.items()}
    namespace.update({func: getattr(np, func) for func in ALLOWED_NUMPY_FUNCS})

    with np.errstate(divide="ignore", invalid="ignore"):
        result = _safe_eval_band_math(expression, namespace)

    result = np.asarray(result, dtype=np.float64)

    # 替换 inf 为 NaN
    result = np.where(np.isfinite(result), result, np.nan)

    return result


def get_preset_index(name: str) -> Optional[Dict]:
    """按名称获取预设指数定义"""
    for p in PRESET_INDICES:
        if p["name"].lower() == name.lower():
            return p
    return None


def compute_index_stats(index_array: np.ndarray, pixel_size_m: float = 10.0) -> Dict:
    """
    计算指数统计信息

    参数:
        index_array: (H, W) 指数数组
        pixel_size_m: 像元大小 (m)

    返回:
        dict: {mean, std, min, max, p5, p95, valid_ratio}
    """
    arr = np.asarray(index_array, dtype=np.float64)
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "p5": None, "p95": None, "valid_ratio": 0.0}

    return {
        "mean": round(float(np.nanmean(valid)), 4),
        "std": round(float(np.nanstd(valid)), 4),
        "min": round(float(np.nanmin(valid)), 4),
        "max": round(float(np.nanmax(valid)), 4),
        "p5": round(float(np.nanpercentile(valid, 5)), 4),
        "p95": round(float(np.nanpercentile(valid, 95)), 4),
        "valid_ratio": round(len(valid) / arr.size, 4),
    }
