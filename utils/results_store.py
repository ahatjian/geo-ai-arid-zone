"""
结果持久化模块 — 分析结果统一存储与检索
==========================================
为平台提供跨会话的结果持久化能力：各分析模块产出的
分类图、统计表、矢量、图表等结果可保存到磁盘，
供「数据下载中心」统一管理、打包下载、跨会话复用。

存储布局:
    BASE_DIR/results/
        <result_id>/            # id = YYYYMMDD_HHMMSS_<rand>
            meta.json           # 元数据 (name/kind/created_at/size/meta)
            data.<ext>          # 实际数据文件

设计约定:
  - 所有保存函数返回 result 元数据 dict
  - 结果 id 全局唯一, 用于后续加载/删除/打包
  - 支持的数据类型 kind: npy/csv/geojson/json/kml/txt/md/png/jpg/tif/zip/pdf
  - 元数据 meta 可附带任意自定义字段 (研究区/日期/参数等)

依赖: numpy, pandas (按 kind 惰性导入)
"""

import os
import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np

from config import BASE_DIR

# ============================================
# 路径与常量
# ============================================

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# kind → 扩展名映射
KIND_EXT = {
    "npy": "npy",
    "csv": "csv",
    "geojson": "geojson",
    "json": "json",
    "kml": "kml",
    "txt": "txt",
    "md": "md",
    "png": "png",
    "jpg": "jpg",
    "tif": "tif",
    "geotiff": "tif",
    "zip": "zip",
    "pdf": "pdf",
}

# 文本类 kind (写入时用 UTF-8)
TEXT_KINDS = {"geojson", "json", "kml", "txt", "md"}
# 二进制类 kind (写入时用 wb)
BINARY_KINDS = {"png", "jpg", "tif", "zip", "pdf"}


def _ensure_dir() -> str:
    """确保结果目录存在并返回路径"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _new_result_id() -> str:
    """生成唯一结果 id (时间戳 + 随机后缀)"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = np.random.randint(1000, 9999)
    return f"{ts}_{suffix}"


def _human_size(num_bytes: int) -> str:
    """字节数 → 人类可读大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


# ============================================
# 保存
# ============================================

def save_result(
    name: str,
    data: Any,
    kind: str = "npy",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    保存一个分析结果到结果库。

    参数:
        name: 结果名称 (展示用)
        data: 结果数据, 类型取决于 kind:
          - npy: numpy 数组
          - csv: pandas DataFrame
          - geojson/json/kml/txt/md: 文本 (str)
          - png/jpg/tif/zip/pdf: 二进制 (bytes)
        kind: 数据类型 (见 KIND_EXT)
        meta: 附加元数据 dict (研究区/日期/参数等), 会被持久化

    返回:
        result dict: {id, name, kind, path, size_bytes, size_human, created_at, meta}
    """
    _ensure_dir()

    if kind not in KIND_EXT:
        raise ValueError(f"不支持的 kind '{kind}', 可选: {list(KIND_EXT.keys())}")

    result_id = _new_result_id()
    result_dir = os.path.join(RESULTS_DIR, result_id)
    os.makedirs(result_dir, exist_ok=True)

    ext = KIND_EXT[kind]
    data_path = os.path.join(result_dir, f"data.{ext}")

    # 按 kind 序列化
    if kind == "npy":
        np.save(data_path, np.asarray(data))
    elif kind == "csv":
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(data_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(data).to_csv(data_path, index=False, encoding="utf-8-sig")
    elif kind in TEXT_KINDS:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(str(data))
    elif kind in BINARY_KINDS:
        with open(data_path, "wb") as f:
            f.write(data if isinstance(data, bytes) else bytes(data))
    else:
        # 兜底: 按文本写入
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(str(data))

    size_bytes = os.path.getsize(data_path)
    # 微秒精度, 保证同一秒内连续保存的结果排序稳定
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    result = {
        "id": result_id,
        "name": name,
        "kind": kind,
        "ext": ext,
        "path": data_path,
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
        "created_at": created_at,
        "meta": meta or {},
    }

    # 写元数据
    meta_path = os.path.join(result_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def save_result_file(
    file_bytes: bytes,
    filename: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    保存任意文件到结果库 (扩展名从 filename 推断)。

    参数:
        file_bytes: 文件二进制内容
        filename: 文件名 (如 "ndvi_map.tif", "stats.csv")
        meta: 附加元数据

    返回:
        result dict (同 save_result)
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"

    # 反向映射 ext → kind
    kind = "bin"
    for k, e in KIND_EXT.items():
        if e == ext:
            kind = k
            break

    result_id = _new_result_id()
    result_dir = os.path.join(_ensure_dir(), result_id)
    os.makedirs(result_dir, exist_ok=True)

    # 保留原始文件名
    safe_name = "".join(c for c in filename if c not in '\\/:*?"<>|') or f"data.{ext}"
    data_path = os.path.join(result_dir, safe_name)

    with open(data_path, "wb") as f:
        f.write(file_bytes)

    size_bytes = os.path.getsize(data_path)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    result = {
        "id": result_id,
        "name": filename,
        "kind": kind,
        "ext": ext,
        "path": data_path,
        "size_bytes": size_bytes,
        "size_human": _human_size(size_bytes),
        "created_at": created_at,
        "meta": meta or {},
    }

    meta_path = os.path.join(result_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================
# 检索
# ============================================

def list_results() -> List[Dict[str, Any]]:
    """列出所有已保存结果 (按时间降序)"""
    _ensure_dir()

    results = []
    for dir_name in os.listdir(RESULTS_DIR):
        result_dir = os.path.join(RESULTS_DIR, dir_name)
        meta_path = os.path.join(result_dir, "meta.json")
        if not os.path.isdir(result_dir) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["id"] = dir_name
            results.append(result)
        except Exception:
            continue

    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def get_result(result_id: str) -> Optional[Dict[str, Any]]:
    """按 id 获取单个结果元数据"""
    meta_path = os.path.join(RESULTS_DIR, result_id, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    result["id"] = result_id
    return result


def get_result_path(result_id: str) -> Optional[str]:
    """按 id 获取结果数据文件路径"""
    result = get_result(result_id)
    return result["path"] if result and os.path.exists(result["path"]) else None


# ============================================
# 加载
# ============================================

def load_result(result_id: str) -> Optional[Any]:
    """
    加载结果数据 (按 kind 反序列化)。

    返回:
        - npy: numpy 数组
        - csv: pandas DataFrame
        - 文本类: str
        - 二进制类: bytes
    """
    result = get_result(result_id)
    if result is None:
        return None

    path = result["path"]
    if not os.path.exists(path):
        return None

    kind = result.get("kind", "bin")
    if kind == "npy":
        return np.load(path)
    elif kind == "csv":
        import pandas as pd
        return pd.read_csv(path)
    elif kind in TEXT_KINDS:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        with open(path, "rb") as f:
            return f.read()


# ============================================
# 删除 / 清空
# ============================================

def delete_result(result_id: str) -> bool:
    """删除单个结果 (含目录)"""
    result_dir = os.path.join(RESULTS_DIR, result_id)
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
        return True
    return False


def clear_results() -> int:
    """清空全部结果, 返回删除的数量"""
    results = list_results()
    for r in results:
        delete_result(r["id"])
    return len(results)


# ============================================
# 打包下载
# ============================================

def package_results(
    result_ids: List[str],
    output_path: Optional[str] = None,
) -> tuple:
    """
    将选中的结果打包为 zip。

    参数:
        result_ids: 要打包的结果 id 列表
        output_path: 输出 zip 路径 (默认临时文件)

    返回:
        (zip 路径, zip 二进制内容)
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        output_path = tmp.name
        tmp.close()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rid in result_ids:
            result = get_result(rid)
            if result is None:
                continue
            data_path = result["path"]
            if not os.path.exists(data_path):
                continue

            # 每个结果放在以名称命名的子目录, 附 meta.json
            safe_name = "".join(c for c in result["name"] if c not in '\\/:*?"<>|') or rid
            arc_dir = f"{safe_name}"
            zf.write(data_path, arcname=f"{arc_dir}/{os.path.basename(data_path)}")

            meta_json = json.dumps(result, ensure_ascii=False, indent=2)
            zf.writestr(f"{arc_dir}/meta.json", meta_json)

    with open(output_path, "rb") as f:
        binary = f.read()

    return output_path, binary


# ============================================
# 概要信息
# ============================================

def get_store_info() -> Dict[str, Any]:
    """返回结果库概要 (结果数 / 占用空间 / 目录)"""
    results = list_results()
    total_bytes = sum(r.get("size_bytes", 0) for r in results)
    return {
        "result_count": len(results),
        "total_bytes": total_bytes,
        "total_human": _human_size(total_bytes),
        "dir": RESULTS_DIR,
    }
