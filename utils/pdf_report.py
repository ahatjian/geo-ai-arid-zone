"""
PDF 报告生成模块 — 科研报告 PDF 导出
====================================
将分析报告数据渲染为带中文排版的 PDF 文档。

基于 reportlab (已安装), 使用系统中文字体 (微软雅黑/黑体)。
适合: 报告导出页一键下载 PDF 版科研报告。

依赖: reportlab (pip install reportlab)
"""

import os
from typing import Dict, List, Optional
from io import BytesIO


def _find_chinese_font() -> Optional[str]:
    """查找系统中文字体路径 (Windows)。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\Deng.ttf",      # 等线
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Linux/macOS 常见路径
    for path in [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if os.path.exists(path):
            return path
    return None


def generate_report_pdf(
    title: str,
    meta_lines: List[str],
    sections: List[Dict],
    footer: str = "",
) -> Optional[bytes]:
    """
    生成中文 PDF 报告。

    参数:
        title: 报告标题
        meta_lines: 元信息行列表 (如 ["研究区域: 塔里木盆地", "分析日期: 2025-06-01"])
        sections: 章节列表, 每项:
            {"heading": "章节标题", "content": "段落文本" | "table": [(标题行, [数据行...]), ...]}
        footer: 页脚文本

    返回:
        bytes: PDF 二进制 (失败返回 None)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable,
        )
    except ImportError:
        return None

    font_path = _find_chinese_font()
    if not font_path:
        return None

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.ttfonts import TTFontFile

        font_name = "CJKFont"
        if font_path.endswith(".ttc"):
            # .ttc 集合需要指定子字体索引 (0=常规)
            from reportlab.pdfbase.ttfonts import TTFontFile as _TTF
            ttffile = _TTF(font_path, subfontIndex=0)
            pdfmetrics.registerFont(TTFont(font_name, ttffile.fileName))
        else:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception:
        # 字体注册失败则回退 (可能无法渲染中文)
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception:
            return None

    # ---- 样式 ----
    title_style = ParagraphStyle(
        "Title", fontName=font_name, fontSize=20, leading=28,
        alignment=TA_CENTER, spaceAfter=6, textColor=colors.HexColor("#1a5276"),
    )
    meta_style = ParagraphStyle(
        "Meta", fontName=font_name, fontSize=10, leading=16,
        alignment=TA_CENTER, textColor=colors.HexColor("#666666"),
    )
    heading_style = ParagraphStyle(
        "Heading", fontName=font_name, fontSize=14, leading=20,
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1a5276"),
    )
    body_style = ParagraphStyle(
        "Body", fontName=font_name, fontSize=10.5, leading=18,
        alignment=TA_LEFT, spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "Footer", fontName=font_name, fontSize=8, leading=12,
        alignment=TA_CENTER, textColor=colors.HexColor("#999999"),
    )

    # ---- 文档 ----
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=title,
    )

    story = [
        Paragraph(title, title_style),
        Spacer(1, 4),
    ]
    for line in meta_lines:
        story.append(Paragraph(line, meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2ecc71")))
    story.append(Spacer(1, 8))

    for sec in sections:
        heading = sec.get("heading", "")
        if heading:
            story.append(Paragraph(heading, heading_style))

        content = sec.get("content", "")
        if content:
            # 分段渲染
            for para in str(content).split("\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(para, body_style))

        table = sec.get("table")
        if table:
            header, rows = table
            data = [header] + list(rows)
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), font_name),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4e6f1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f8ff")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 4))

    if footer:
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
        story.append(Spacer(1, 4))
        story.append(Paragraph(footer, footer_style))

    doc.build(story)
    return buf.getvalue()


def report_to_pdf_download(
    title: str,
    meta_lines: List[str],
    sections: List[Dict],
    footer: str = "",
) -> Optional[bytes]:
    """报告 → PDF 字节 (供 st.download_button 使用)。"""
    return generate_report_pdf(title, meta_lines, sections, footer)
