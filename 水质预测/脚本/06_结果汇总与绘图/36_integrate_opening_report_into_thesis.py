"""以当前论文稿为底稿，补入开题报告中可用于论文的内容并修复第 5 章空图。"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm, RGBColor
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "文档" / "论文材料" / "福州大学格式论文草稿_香港近岸多水层水质预测_硕士论文扩展审阅版_20260907.docx"
OUTPUT = ROOT / "文档" / "论文材料" / "福州大学格式论文草稿_香港近岸多水层水质预测_开题内容融合与图件修订版_v2_20260907.docx"
FIGURE_DIR = ROOT / "结果" / "论文图表" / "遥感字段核对_20260907"


def find_paragraph(document: Document, starts_with: str):
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(starts_with):
            return paragraph
    raise KeyError(f"未找到段落：{starts_with}")


def set_black(paragraph) -> None:
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)


def replace_paragraph(paragraph, text: str) -> None:
    style = paragraph.style
    paragraph.text = text
    paragraph.style = style
    set_black(paragraph)


def insert_before_anchor(anchor, text: str, style_name: str):
    paragraph = anchor.insert_paragraph_before(text, style=style_name)
    set_black(paragraph)
    return paragraph


def insert_after_paragraph(paragraph, text: str, style_name: str):
    element = OxmlElement("w:p")
    paragraph._p.addnext(element)
    inserted = Paragraph(element, paragraph._parent)
    inserted.style = style_name
    inserted.add_run(text)
    set_black(inserted)
    return inserted


def insert_picture_after(paragraph, image_path: Path, width_cm: float = 15.0) -> None:
    inserted = insert_after_paragraph(paragraph, "", "Normal")
    inserted.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inserted.add_run().add_picture(str(image_path), width=Cm(width_cm))
    return inserted


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    for name in ["图5-5_遥感字段进入正式模型的情况.png", "图5-6_当前遥感特征解释的范围.png"]:
        if not (FIGURE_DIR / name).exists():
            raise FileNotFoundError(FIGURE_DIR / name)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="wq_thesis_opening_", dir=OUTPUT.parent))
    working_copy = temp_root / "current_thesis.docx"
    checked_output = temp_root / "updated_thesis.docx"
    shutil.copy2(SOURCE, working_copy)

    document = Document(working_copy)
    normal_style = find_paragraph(document, "研究区覆盖香港近岸").style.name
    heading_style = find_paragraph(document, "1.4 本文主要工作").style.name

    chapter_two = find_paragraph(document, "第2章 研究区")
    insert_before_anchor(chapter_two, "1.5 研究基础与本文定位", heading_style)
    insert_before_anchor(
            chapter_two,
            "本节补充本文的研究基础。主要说明研究区特点和本文采用多源对照的原因。它把第1章的问题与第2章的数据连接起来。",
            normal_style,
    )
    insert_before_anchor(
            chapter_two,
            "香港近岸同时受到珠江径流、潮汐交换、天气变化和城市活动影响。港内、半封闭海湾和外海的水体条件不同，因此同一套预测方法在不同水区和水层中可能得到不同结果。",
            normal_style,
    )
    insert_before_anchor(
            chapter_two,
            "本文沿用多源水质预测的研究思路，但不把遥感视为单独的反演结果。遥感、气象、降雨、潮汐和历史信息分别进入对照组，用来判断它们在不同指标和水层中的作用。",
            normal_style,
    )

    study_area = find_paragraph(document, "研究区覆盖香港近岸")
    added = insert_after_paragraph(
            study_area,
            "香港位于珠江口东侧。近岸水域包括港内、半封闭海湾和外海。不同水域的潮汐交换和水体滞留时间不同，这为比较分区和全域模型提供了现实背景。",
            normal_style,
    )
    insert_after_paragraph(
            added,
            "沿岸城市活动、港口航运和陆源输入会改变局地水质。雨季径流增强时，这种差异更明显。因此，本文同时保留水区、季节和水层信息，并把它们用于结果解释。",
            normal_style,
    )

    heading = find_paragraph(document, "5.2 遥感特征可解释性")
    replace_paragraph(heading, "5.2 当前正式主表中的遥感特征说明")
    explanation = find_paragraph(document, "采用 SHAP 解释")
    replace_paragraph(
            explanation,
            "当前正式主表保存了 Landsat 原始波段、指数和可用性标记。当前训练脚本需要的高质量波段统计没有完整写入正式训练表，因此模型不能给出波段级 SHAP 结果。本节只说明遥感字段的进入情况和当前可以讨论的范围。",
    )

    caption_55 = find_paragraph(document, "图5-5 实际可用遥感特征")
    replace_paragraph(caption_55, "图5-5 当前正式主模型的遥感字段进入情况")
    insert_picture_after(caption_55, FIGURE_DIR / "图5-5_遥感字段进入正式模型的情况.png")

    caption_56 = find_paragraph(document, "图5-6 实际可用遥感主要特征")
    replace_paragraph(caption_56, "图5-6 当前遥感特征解释的范围")
    insert_picture_after(caption_56, FIGURE_DIR / "图5-6_当前遥感特征解释的范围.png")

    document.save(checked_output)
    with zipfile.ZipFile(checked_output) as package:
        bad_member = package.testzip()
    if bad_member is not None:
        raise RuntimeError(f"生成的 DOCX 损坏：{bad_member}")
    os.replace(checked_output, OUTPUT)
    with zipfile.ZipFile(OUTPUT) as package:
        bad_member = package.testzip()
    if bad_member is not None:
        raise RuntimeError(f"替换后的 DOCX 损坏：{bad_member}")
    shutil.rmtree(temp_root, ignore_errors=True)
    print(f"输出文件：{OUTPUT}")


if __name__ == "__main__":
    main()
