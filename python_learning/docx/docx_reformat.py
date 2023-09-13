from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.text.paragraph import Paragraph
from docx.shape import InlineShape
import docx
import os
from copy import deepcopy
import docx2txt
from docx.shared import Cm, Inches
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_UNDERLINE
from docx.shared import RGBColor, Pt, Cm, Inches
from docx.enum.style import WD_STYLE_TYPE
from docx.text.parfmt import ParagraphFormat
from utils import set_cell_border
from docx.styles.style import _ParagraphStyle as ParagraphStyle
from utils import (
    getParagraphRuns,
    get_word_image_paths,
    delete_paragraph,
    iter_block_items,
)

path = "/home/bocheng/data/pdf/document_parse/Word/random/atmosphere-2489382.docx"

# 提取word文档中的图片,图片格式的支持需要手动添加
image_dir = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data"
docx2txt.process(path, image_dir)

# path = "data/inplace_demo.docx"
src_doc = docx.Document(path)
dst_doc = deepcopy(src_doc)

# 将所有段落文本的字体进行统一设置：主要是为了解决超链接字体设置问题
for p in dst_doc.paragraphs:
    runs = getParagraphRuns(p)
    # runs = getHyperlinksRuns(p)
    for run in runs:
        run.font.name = "Palatino Linotype"
        run.font.size = Pt(10)

# 页眉样式
page_header_style = dst_doc.styles.add_style("textstyle", WD_STYLE_TYPE.PARAGRAPH)
page_header_style.font.size = Pt(8)  # 样式字体大小

image_name_path = get_word_image_paths(image_dir)

rels = {}
for r in src_doc.part.rels.values():
    if isinstance(r._target, docx.parts.image.ImagePart):
        rels[r.rId] = os.path.basename(r._target.partname)
image_id = 1

reference_index = -1

for i, block in enumerate(iter_block_items(dst_doc)):
    # 如果是段落
    if isinstance(block, Paragraph):
        # Read and process the Paragraph
        # If you find an image
        # 如果包含图片
        if "Graphic" in block._p.xml:
            for rId in rels:
                if rId in block._p.xml:
                    # Your image will be in os.path.join(img_path, rels[rId])
                    image_path = image_name_path.get(f"image{image_id}", None)
                    if image_path is not None:
                        pic_runs = block.runs
                        for pic_run in block.runs:
                            pic_run.clear()
                        pic = pic_runs[-1].add_picture(
                            open(image_path, mode="rb"),
                        )
                        # 设置图片宽高
                        pic.width = Cm(8)
                        pic.height = Cm(6)
                    image_id += 1
                # 设置图片对齐
                block.alignment = WD_ALIGN_PARAGRAPH.LEFT  # 段落左对齐
                block.paragraph_format.left_indent = Cm(4.6)  # 段落左侧间距
                block.paragraph_format.line_spacing = 0.95  # 段落行间距
                block.paragraph_format.space_before = 0  # 段前距离
                block.paragraph_format.space_after = 0  # 段后距离
        # 如果是不包含图片的段落
        else:
            # 如果段落不为空
            if block.text.strip():
                # 段落是一级标题
                if "heading 1" in block.style.name.lower():
                    head1 = block
                    head1.style.font.size = Pt(10)
                    head1.style.font.bold = True
                    head1.style.font.name = "Palatino Linotype"
                    head1.paragraph_format.line_spacing = 0.95
                    head1.paragraph_format.left_indent = 0
                    # 判断是否是Reference
                    if (
                        "reference" in block.text.lower()
                        and len(block.text.strip()) < 12
                    ):
                        reference_index = i
                        head1.paragraph_format.space_after = 0
                        head1.paragraph_format.space_before = 0
                        head1.paragraph_format.first_line_indent = 0
                    else:
                        head1.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        head1.paragraph_format.left_indent = Cm(4.6)
                        head1.paragraph_format.space_after = Pt(3)
                        head1.paragraph_format.space_before = Pt(12)
                # 段落是二级标题
                elif "heading 2" in block.style.name.lower():
                    if (
                        "reference" in block.text.lower()
                        and len(block.text.strip()) < 12
                    ):
                        reference_index = i
                    head2 = block
                    head2.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    head2.style.font.size = Pt(10)
                    head2.style.font.italic = True
                    head2.style.font.name = "Palatino Linotype"
                    head2.paragraph_format.left_indent = Cm(4.6)
                    head2.paragraph_format.line_spacing = 0.95
                    head2.paragraph_format.space_after = Pt(3)
                    head2.paragraph_format.space_before = Pt(12)
                # 段落是三级标题
                elif "heading 3" in block.style.name.lower():
                    head3 = block
                    head3.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    head3.style.font.size = Pt(10)
                    head3.style.font.name = "Palatino Linotype"
                    head3.paragraph_format.left_indent = Cm(4.6)
                    head3.paragraph_format.line_spacing = 0.95
                    head3.paragraph_format.space_before = Pt(12)
                    head3.paragraph_format.space_after = Pt(3)
                # 段落是正文
                else:
                    block.style.font.size = Pt(10)  # 样式字体大小
                    block.style.font.name = "Palatino Linotype"
                    block.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY  # 段落右对齐
                    block.paragraph_format.first_line_indent = Cm(0.75)  # 首行缩进两个字符
                    block.paragraph_format.left_indent = Cm(4.6)
                    block.paragraph_format.line_spacing = 0.95
                    block.paragraph_format.space_before = 0
                    block.paragraph_format.space_after = 0
                    for run in block.runs:
                        run.font.name = "Palatino Linotype"
                        run.font.size = Pt(10)
                    if (
                        reference_index == -1
                        and "reference" in block.text.lower()
                        and len(block.text.strip()) < 12
                    ):
                        reference_index = i
                    elif reference_index != -1:
                        # 参考文献样式设置
                        if i > reference_index:  # text after references
                            for run in block.runs:
                                run.font.name = "Palatino Linotype"
                                run.font.size = Pt(9)
                            block.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY  # 段落右对齐
                            block.paragraph_format.first_line_indent = Cm(
                                0.0
                            )  # 首行缩进两个字符
                            block.paragraph_format.left_indent = 0
                            block.paragraph_format.line_spacing = 0.95
                            block.paragraph_format.space_before = 0
                            block.paragraph_format.space_after = 0
            else:
                # 空段落则清除掉
                block.clear()
                delete_paragraph(block)
    # 表格
    elif isinstance(block, Table):
        block.alignment = WD_TABLE_ALIGNMENT.CENTER
        block.style.paragraph_format.alignment = WD_TABLE_ALIGNMENT.CENTER
        block.style.font.size = Pt(10)
        rows = block.rows
        columns = block.columns
        for r in range(len(rows)):
            for c in range(len(columns)):
                cell_one = block.cell(r, c)
                if r == 0:
                    set_cell_border(
                        cell=cell_one,
                        top={
                            "sz": 8,
                            "val": "single",
                            "color": "000000",
                            "space": "0",
                        },  # 8-> 4pt
                        bottom={"sz": 0.5, "color": "#000000", "val": "single"},
                        left={"sz": 24, "val": "none"},
                        right={"sz": 12, "val": "none"},
                    )
                elif r == len(rows) - 1:
                    set_cell_border(
                        cell=cell_one,
                        top={
                            "sz": 0.5,
                            "val": "single",
                            "color": "000000",
                            "space": "0",
                        },
                        bottom={"sz": 8, "color": "#000000", "val": "single"},
                        left={"sz": 24, "val": "none"},
                        right={"sz": 12, "val": "none"},
                    )
                else:
                    set_cell_border(
                        cell=cell_one,
                        top={
                            "sz": 0.5,
                            "val": "single",
                            "color": "000000",
                            "space": "0",
                        },
                        bottom={"sz": 0.5, "color": "#000000", "val": "single"},
                        left={"sz": 24, "val": "none"},
                        right={"sz": 12, "val": "none"},
                    )
                for p in cell_one.paragraphs:
                    # 设置表格中超链接的样式
                    runs = getParagraphRuns(p)
                    for run in runs:
                        run.font.name = "Palatino Linotype"
                        run.font.size = Pt(10)
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for cell_run in p.runs:
                        cell_run.font.name = "Palatino Linotype"
                        cell_run.font.size = Pt(10)
    elif isinstance(block, InlineShape):
        # Read and process the Picture
        print("Picture found:", block)


def add_page_header_footer(dst_doc: docx.Document) -> docx.Document:
    dst_doc.add_section(
        start_type=WD_SECTION.CONTINUOUS
    )  ###通过在页眉添加表格，在表格的单元中插入图片的方式实现在页眉插入多张图片，并两端对齐
    image_header1 = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header.png"
    image_header2 = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header_02.png"
    sections = dst_doc.sections
    section0 = sections[0]
    # print("页面上边距：", section0.top_margin)
    # print("页面宽度：", section0.page_width)

    # 设置页面距离：上下左右
    section0.top_margin = Cm(2.5)
    section0.bottom_margin = Cm(1.9)
    section0.left_margin = Cm(1.27)
    section0.right_margin = Cm(1.27)

    # 设置页眉页脚的上下距离
    section0.header_distance = Cm(1.8)
    section0.footer_distance = Cm(0.6)
    section0.different_first_page_header_footer = True  # 首页页眉页脚不同
    section_0_header = section0.first_page_header  # 首页页眉
    # 页眉添加两张图片及样式设置
    section_0_header.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    section0_run0 = section_0_header.paragraphs[0].add_run()
    #
    section0_run0.add_picture(image_header1, width=Cm(4.64), height=Cm(1.21))
    spaces = [" "] * 141
    section0_run0.add_text("".join(spaces))
    section0_run0.add_picture(image_header2, width=Cm(1.5), height=Cm(1))
    # section0_run0.underline = WD_UNDERLINE.SINGLE
    # section_0_header.paragraphs[0].add_run("".join(spaces))
    # section0_run1 = section_0_header.paragraphs[0].add_run()
    # section0_run1.add_picture(image_header2, width=Cm(1.5), height=Cm(1))
    underline_para = section_0_header.add_paragraph()
    underline_para.paragraph_format.line_spacing = Pt(5)
    underline_run = underline_para.add_run()
    underline_run.font.size = Pt(0.5)
    underline_run.add_tab()
    underline_run.add_tab()
    underline_run.add_tab()
    underline_run.add_tab()
    underline_run.add_tab()
    underline_run.add_tab()
    underline_run.add_tab()

    underline_run.underline = WD_UNDERLINE.SINGLE

    # 首页页脚，通过表格的方式添加左右两边的内容，方便对齐
    section_0_footer = section0.first_page_footer

    footer_para = section_0_footer.add_paragraph(
        "Int. J. Mol. Sci. 2023, 24, x. https://doi.org/10.3390/xxxxx",
        style=page_header_style,
    )
    footer_para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    tab_stops = footer_para.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(25), alignment=WD_TAB_ALIGNMENT.RIGHT)
    footer_para.add_run().add_tab()
    footer_para.add_run("www.mdpi.com/journal/ijms")

    htable = section_0_footer.add_table(1, 2, width=Cm(21))
    htable.rows[0].height = Cm(0.5)
    htab_cells = htable.rows[0].cells
    ht0 = htab_cells[0].add_paragraph(
        "Int. J. Mol. Sci. 2023, 24, x. https://doi.org/10.3390/xxxxx",
        style=page_header_style,
    )
    ht0.alignment = WD_ALIGN_PARAGRAPH.LEFT

    ht1 = htab_cells[1].add_paragraph(
        "www.mdpi.com/journal/ijms",
        style=page_header_style,
    )
    ht1.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # 控制后续的页眉页脚跟前面的不一样
    sections[1].header_distance = Cm(1.8)
    sections[1].footer_distance = Cm(0.6)
    section_1_header = sections[1].header
    section_1_header.paragraphs[0].add_run(
        "Int. J. Mol. Sci. 2023, 24, x FOR PEER REVIEW"
    )

    section_1_header.is_linked_to_previous = False

    section_1_footer = sections[1].footer
    section_1_footer.is_linked_to_previous = False
    return dst_doc


dst_doc = add_page_header_footer(dst_doc)
dst_doc.save("data/inplace_demo.docx")
