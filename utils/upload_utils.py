"""
上传文件处理工具 — 稳定临时路径 (防磁盘泄漏)
==============================================
Streamlit 的 rerun 机制下, 每次重跑都会重新执行上传保存逻辑。
若用 uuid/时间戳命名, 每次 rerun 都生成新文件 → 无限积累磁盘泄漏。

本模块用"内容 hash"命名: 同一文件反复上传 → 同一路径 (覆盖),
不同文件 → 不同路径。既防路径注入 (不信任客户端文件名做路径),
又防泄漏 (文件数有界)。

用法:
    from utils.upload_utils import save_upload_stable
    path = save_upload_stable(uploaded, "spatial")
"""

import os
import hashlib
import tempfile


def save_upload_stable(uploaded, prefix: str, max_size_mb: int = 200) -> str:
    """
    保存上传文件到稳定的临时路径 (内容 hash 命名)。

    参数:
        uploaded: Streamlit UploadedFile
        prefix: 文件名前缀 (如 "spatial" / "atmo" / "veg")
        max_size_mb: 大小上限 (MB), 防超大文件

    返回:
        str: 临时文件路径
    """
    content = uploaded.getvalue()
    if len(content) > max_size_mb * 1024 * 1024:
        raise ValueError(f"文件过大 (>{max_size_mb}MB)")

    # 内容 hash (前 256KB 足够区分, 速度快)
    h = hashlib.md5(content[:262144]).hexdigest()[:12]

    # 扩展名白名单 (不信任客户端)
    ext = os.path.splitext(getattr(uploaded, "name", "") or "")[1].lower()
    if ext not in (".tif", ".tiff", ".geojson", ".json", ".csv", ".npz", ".npy"):
        ext = ".tif"

    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{h}{ext}")
    # 已存在且大小一致 → 跳过写入 (避免重复 IO)
    if os.path.exists(path) and os.path.getsize(path) == len(content):
        return path

    with open(path, "wb") as f:
        f.write(content)
    return path


def cleanup_old_uploads(prefix: str, keep: int = 20) -> int:
    """
    清理旧上传文件 (按修改时间保留最近 keep 个)。

    参数:
        prefix: 文件名前缀
        keep: 保留数量

    返回:
        int: 删除文件数
    """
    tmp = tempfile.gettempdir()
    try:
        files = [
            os.path.join(tmp, f)
            for f in os.listdir(tmp)
            if f.startswith(f"{prefix}_") and f.endswith((".tif", ".tiff", ".geojson", ".json"))
        ]
    except OSError:
        return 0

    if len(files) <= keep:
        return 0

    files.sort(key=os.path.getmtime, reverse=True)
    removed = 0
    for f in files[keep:]:
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    return removed
