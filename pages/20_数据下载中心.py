"""
数据下载中心页面 — 结果持久化与统一下载
==========================================
管理平台各分析模块产出的结果，提供跨会话持久化、
统一检索、预览、单文件下载、批量打包 (zip) 与删除能力。

功能:
  - 结果入库 (上传任意文件保存)
  - 结果列表 (按时间倒序, 复选框多选)
  - 单文件下载 (按类型匹配 MIME)
  - 批量打包下载 (zip)
  - 删除选中 / 清空全部
  - 文本/CSV/npy/图片结果预览
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.results_store import (
    save_result_file, list_results, get_result, get_result_path,
    load_result, delete_result, clear_results, package_results, get_store_info,
)
from utils.error_handler import StreamlitErrorBoundary

st.set_page_config(page_title="数据下载中心", page_icon="📦", layout="wide")

# ============================================================
# 常量: 扩展名 → MIME 类型
# ============================================================
MIME_MAP = {
    "npy": "application/octet-stream",
    "csv": "text/csv",
    "geojson": "application/geo+json",
    "json": "application/json",
    "kml": "application/vnd.google-earth.kml+xml",
    "txt": "text/plain",
    "md": "text/markdown",
    "png": "image/png",
    "jpg": "image/jpeg",
    "tif": "image/tiff",
    "zip": "application/zip",
    "pdf": "application/pdf",
    "bin": "application/octet-stream",
}

# 文本类扩展名 (可预览)
TEXT_EXTS = {"geojson", "json", "kml", "txt", "md"}
# 图片类扩展名 (可预览)
IMAGE_EXTS = {"png", "jpg"}


def _mime_for(ext: str) -> str:
    """按扩展名取 MIME 类型"""
    return MIME_MAP.get(ext, "application/octet-stream")


def _download_filename(result: dict) -> str:
    """构造下载文件名 (含正确扩展名)"""
    name = result.get("name", result.get("id", "result"))
    ext = result.get("ext", "bin")
    # 名称已含扩展名则直接用, 否则补全
    if "." in name and name.rsplit(".", 1)[-1].lower() == ext:
        return name
    return f"{name}.{ext}"


def _render_preview(result: dict):
    """按类型渲染结果预览"""
    kind = result.get("kind", "bin")
    ext = result.get("ext", "bin")

    with st.expander(f"🔍 预览: {result['name']}", expanded=False):
        try:
            if kind == "csv":
                data = load_result(result["id"])
                if data is not None:
                    st.dataframe(data, use_container_width=True)
                return

            if kind == "npy":
                data = load_result(result["id"])
                if data is not None:
                    arr = np.asarray(data)
                    st.markdown(
                        f"**形状**: `{arr.shape}` | **dtype**: `{arr.dtype}` | "
                        f"**min**: `{arr.min():.4f}` | **max**: `{arr.max():.4f}`"
                    )
                return

            if ext in IMAGE_EXTS:
                path = get_result_path(result["id"])
                if path:
                    st.image(path, use_container_width=True)
                return

            if ext in TEXT_EXTS:
                data = load_result(result["id"])
                if data is not None:
                    text = data if isinstance(data, str) else data.decode("utf-8")
                    # 超长文本截断展示
                    if len(text) > 5000:
                        st.text(text[:5000] + "\n\n... (内容过长，已截断，请下载查看完整内容)")
                    else:
                        st.text(text)
                return

            # 二进制文件 (tif/zip/pdf/bin): 仅显示文件信息
            st.info(
                f"二进制文件 ({ext.upper()})，大小 {result.get('size_human', '?')}，"
                "无法在线预览，请直接下载。"
            )
        except Exception as e:
            st.warning(f"预览失败: {e}")


# ============================================================
# 标题区
# ============================================================
st.title("📦 数据下载中心")
st.markdown(
    "统一管理平台分析结果，支持**跨会话持久化**、单文件下载、"
    "批量打包 (zip) 与删除。各分析模块产出的分类图、统计表、矢量等结果"
    "可在此集中下载，衔接 ArcGIS / QGIS / Excel 等桌面工具。"
)

# ============================================================
# 概要信息
# ============================================================
with StreamlitErrorBoundary("概要信息", st=st, show_traceback=False):
    info = get_store_info()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已保存结果", info["result_count"])
    with col2:
        st.metric("占用空间", info["total_human"])
    with col3:
        st.metric("存储位置", "results/ 目录")

st.divider()

# ============================================================
# 结果入库 (上传保存)
# ============================================================
st.subheader("⬆️ 结果入库")
st.caption("上传任意分析结果文件 (GeoTIFF/CSV/GeoJSON/PNG/PDF 等)，持久化保存到结果库。")

with st.form("save_form", clear_on_submit=True):
    col_file, col_name = st.columns([3, 1])
    with col_file:
        up_file = st.file_uploader(
            "选择文件",
            type=None,  # 不限类型
            key="dc_upload",
            label_visibility="collapsed",
        )
    with col_name:
        note = st.text_input(
            "备注 (可选)",
            placeholder="如: 塔里木 NDVI 2024",
            key="dc_note",
        )
    submitted = st.form_submit_button("💾 保存到结果库", type="primary")

    if submitted:
        if up_file is None:
            st.error("⚠️ 请先选择要保存的文件")
        else:
            try:
                meta = {}
                if note.strip():
                    meta["note"] = note.strip()
                res = save_result_file(up_file.getvalue(), up_file.name, meta=meta)
                st.success(f"✅ 已保存: **{res['name']}** ({res['size_human']})")
                st.rerun()
            except Exception as e:
                st.error(f"保存失败: {e}")

st.divider()

# ============================================================
# 结果列表
# ============================================================
st.subheader("📋 结果列表")

results = list_results()

if not results:
    st.info("📭 结果库为空。可先在上方「结果入库」上传文件，或到各分析模块导出结果后保存。")
    st.stop()

# 构建表格
rows = []
for r in results:
    rows.append({
        "选择": False,
        "ID": r["id"],
        "名称": r["name"],
        "类型": (r.get("ext", "bin") or "bin").upper(),
        "大小": r.get("size_human", "?"),
        "时间": r.get("created_at", "?")[:19],
    })
df = pd.DataFrame(rows)

# 编辑表格 (复选框多选)
edited = st.data_editor(
    df,
    hide_index=True,
    use_container_width=True,
    disabled=["ID", "名称", "类型", "大小", "时间"],
    key="dc_table",
)

selected = [row["ID"] for row in edited.to_dict("records") if row.get("选择")]

# ============================================================
# 操作区
# ============================================================
st.subheader("🛠️ 操作")
st.caption(f"已选中 **{len(selected)}** 个结果。")

col_dl, col_zip, col_del, col_clear = st.columns([1.2, 1.2, 1.2, 1.2])

with col_dl:
    if len(selected) == 1:
        rid = selected[0]
        res = get_result(rid)
        if res:
            try:
                with open(res["path"], "rb") as f:
                    raw = f.read()
                st.download_button(
                    "⬇️ 下载选中",
                    data=raw,
                    file_name=_download_filename(res),
                    mime=_mime_for(res.get("ext", "bin")),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"下载失败: {e}")
    elif len(selected) > 1:
        st.info("多选请用「打包下载」")
    else:
        st.info("勾选结果后下载")

with col_zip:
    if selected:
        try:
            _, zip_binary = package_results(selected)
            st.download_button(
                "🗜️ 打包下载 (zip)",
                data=zip_binary,
                file_name=f"geoai_results_{len(selected)}.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"打包失败: {e}")
    else:
        st.info("勾选结果后打包")

with col_del:
    if selected:
        if st.button("🗑️ 删除选中", use_container_width=True):
            for rid in selected:
                delete_result(rid)
            st.success(f"✅ 已删除 {len(selected)} 个结果")
            st.rerun()
    else:
        st.info("勾选结果后删除")

with col_clear:
    with st.popover("⚠️ 清空全部"):
        st.warning("此操作将删除结果库中**全部**结果，且不可恢复！")
        if st.button("确认清空", type="primary", use_container_width=True):
            n = clear_results()
            st.success(f"✅ 已清空 {n} 个结果")
            st.rerun()

st.divider()

# ============================================================
# 结果明细 + 预览
# ============================================================
st.subheader("🔎 结果明细")
for r in results:
    meta = r.get("meta", {}) or {}
    note = meta.get("note", "") if isinstance(meta, dict) else ""
    meta_desc = f" | 📝 {note}" if note else ""
    st.markdown(
        f"**{r['name']}** — `{r['id']}` | `{(r.get('ext','bin') or 'bin').upper()}` "
        f"| {r.get('size_human','?')} | {r.get('created_at','?')[:19]}{meta_desc}"
    )
    _render_preview(r)

st.divider()
st.info(
    "💡 **提示**: 结果保存在本地 `results/` 目录，跨会话持久化。"
    "各分析模块的「导出」结果可先入库，再在此统一下载或打包。"
    "若需衔接桌面 GIS，推荐将 GeoTIFF / GeoJSON / Shapefile 结果在此集中管理。"
)
