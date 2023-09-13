import docx
from docx.text.run import Run
from docx.oxml.shared import OxmlElement, qn
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
from docx.text.paragraph import Paragraph
from docx.shape import InlineShape
import os
import jieba

style_doc = docx.Document(
    "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/docx/ijms-template.docx"
)


# 设置单元格的边框
def set_cell_border(cell, **kwargs):
    """
    Set cell`s border
    Usage:
    set_cell_border(
        cell,
        top={"sz": 12, "val": "single", "color": "FF0000", "space": "0"},
        bottom={"sz": 12, "color": "00FF00", "val": "single"},
        left={"sz": 24, "val": "dashed", "shadow": "true"},
        right={"sz": 12, "val": "dashed"},
    )
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)

    for edge in ("left", "top", "right", "bottom", "insideH", "insideV"):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = "w:{}".format(edge)

            # check for tag existnace, if none found, then create one
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)

            # looks like order of attributes is important
            for key in ["sz", "val", "color", "space", "shadow"]:
                if key in edge_data:
                    element.set(qn("w:{}".format(key)), str(edge_data[key]))


def delete_paragraph(paragraph):
    p = paragraph._element
    p.getparent().remove(p)
    p._p = p._element = None


def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    elif isinstance(parent, _Row):
        parent_elm = parent._tr
    else:
        raise ValueError("Something is not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)
        elif isinstance(child, InlineShape):
            yield child


def get_style_templates():
    style_dic = {}
    for paragraph in style_doc.paragraphs:
        # print(
        #     paragraph.style,
        #     "\t",
        #     paragraph.style.name,
        #     # "\t",
        #     # paragraph.style.paragraph_format,
        # )
        style_name = paragraph.style.name.split("_")[-1]
        if style_name == "type":
            style_name = "article_type"
        style_dic[style_name] = paragraph.style
    return style_dic


def getHyperlinksRuns(paragraph):
    def _get(node, parent):
        for child in node:
            if child.tag == qn("w:hyperlink"):
                yield from returnRun(child, parent)

    def returnRun(node, parent):
        for child in node:
            if child.tag == qn("w:r"):
                yield Run(child, parent)

    return list(_get(paragraph._element, paragraph))


def getParagraphRuns(paragraph):
    def _get(node, parent):
        for child in node:
            if child.tag == qn("w:r"):
                yield Run(child, parent)
            if child.tag == qn("w:hyperlink"):
                yield from _get(child, parent)

    return list(_get(paragraph._element, paragraph))


def get_word_image_paths(image_dir: str):
    image_name_path = {}
    for image_name in os.listdir(image_dir):
        if image_name.startswith("image"):
            image_name_suffix = os.path.splitext(image_name)[0]
            image_name_path[image_name_suffix] = os.path.join(image_dir, image_name)
    return image_name_path


def jaccard_title(source: str, reference: str):
    """计算两个字符串的jaccard相似度，用于判断是否标题
    :param source: 原始论文稿件
    :param reference: 使用grobid解析出的title
    """
    source_tokens = set([w for w in jieba.cut(source) if w.strip()])
    reference_tokens = set([w for w in jieba.cut(reference) if w.strip()])
    tmp = 0
    for i in reference_tokens:
        if i in source_tokens:
            tmp += 1
    total = len(source_tokens) + len(reference_tokens) - tmp
    try:
        jaccard_similarity = float(tmp / total)
    except ZeroDivisionError:
        print(source, reference)
        return 0
    else:
        return jaccard_similarity


if __name__ == "__main__":
    title = "Material substitution strategies for energy reduction and greenhouse gas emission in cement manufacturing"
    reference = "strategies for energy reduction and greenhouse"
    print(jaccard_title(title, reference))
    # doc = docx.Document()
    # table = doc.add_table(rows=5, cols=4)
    # rows = table.rows
    # columns = table.columns
    # for r in range(len(rows)):
    #     for c in range(len(columns)):
    #         cell_one = table.cell(r, c)
    #         if r == 0:
    #             set_cell_border(
    #                 cell=cell_one,
    #                 top={"sz": 8, "val": "single", "color": "000000", "space": "0"},
    #                 bottom={"sz": 0.5, "color": "#000000", "val": "single"},
    #             )
    #         elif r == len(rows) - 1:
    #             set_cell_border(
    #                 cell=cell_one,
    #                 top={"sz": 0.5, "val": "single", "color": "000000", "space": "0"},
    #                 bottom={"sz": 8, "color": "#000000", "val": "single"},
    #             )
    #         else:
    #             set_cell_border(
    #                 cell=cell_one,
    #                 top={"sz": 0.5, "val": "single", "color": "000000", "space": "0"},
    #                 bottom={"sz": 0.5, "color": "#000000", "val": "single"},
    #             )
    # doc.save("data/test.docx")
