from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.text.paragraph import Paragraph
from docx.shape import InlineShape
import docx
import os
from copy import deepcopy
import docx2txt
from docx.shared import Cm, Inches
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import RGBColor, Pt, Cm, Inches
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.parfmt import ParagraphFormat
from utils import get_style_templates
from docx.styles.style import _ParagraphStyle as ParagraphStyle


path = "/home/bocheng/data/pdf/document_parse/Word/random/atmosphere-2489382.docx"
# path = "data/inplace_demo.docx"
src_doc = docx.Document(path)

dst_doc = deepcopy(src_doc)

# normal_text_style = dst_doc.styles.add_style(
#     "normal_textstyle", WD_STYLE_TYPE.PARAGRAPH
# )
# normal_text_style.font.size = Pt(10)  # 样式字体大小
# normal_text_style.font.name = "Palatino Linotype"


page_header_style = dst_doc.styles.add_style("textstyle", WD_STYLE_TYPE.PARAGRAPH)
page_header_style.font.size = Pt(8)  # 样式字体大小
# page_header_style.font.color.rgb = RGBColor(66, 100, 0)  # 字体颜色


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


image_dir = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data"
docx2txt.process(path, image_dir)


def get_image_paths(image_dir: str):
    image_name_path = {}
    for image_name in os.listdir(image_dir):
        if image_name.startswith("image"):
            image_name_suffix = os.path.splitext(image_name)[0]
            image_name_path[image_name_suffix] = os.path.join(image_dir, image_name)
    return image_name_path


image_name_path = get_image_paths(image_dir)

rels = {}
for r in src_doc.part.rels.values():
    if isinstance(r._target, docx.parts.image.ImagePart):
        rels[r.rId] = os.path.basename(r._target.partname)
image_id = 1

reference_index = -1

for i, block in enumerate(iter_block_items(dst_doc)):
    if isinstance(block, Paragraph):
        # Read and process the Paragraph
        # If you find an image
        if "Graphic" in block._p.xml:
            # new_block = deepcopy(block)
            # Get the rId of the image
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
                        pic.width = Cm(8)
                        pic.height = Cm(6)
                        # pic = new_block.add_run().add_picture(
                        #     open(image_path, mode="rb"), width=Cm(14), height=Cm(10)
                        # )
                        # pic = dst_doc.add_picture(
                        #     open(image_path, mode="rb"),
                        # )
                        # pic.width = Cm(14)
                        # pic.height = Cm(10)
                        # block = new_block
                    image_id += 1
                block.alignment = WD_ALIGN_PARAGRAPH.LEFT  # 段落右对齐
                block.paragraph_format.left_indent = Cm(4.6)
                block.paragraph_format.line_spacing = 0.95
                block.paragraph_format.space_before = 0
                block.paragraph_format.space_after = 0
            # print(f"image {image_path} has been added to dst document")
            # block.alignment = WD_ALIGN_PARAGRAPH.CENTER

        else:
            if block.text.strip():
                if "heading 1" in block.style.name.lower():
                    head1 = block
                    head1.style.font.size = Pt(10)
                    head1.style.font.bold = True
                    head1.style.font.name = "Palatino Linotype"
                    head1.paragraph_format.line_spacing = 0.95
                    head1.paragraph_format.left_indent = 0
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
                elif "heading 2" in block.style.name.lower():
                    if (
                        "reference" in block.text.lower()
                        and len(block.text.strip()) < 12
                    ):
                        reference_index = i
                    # head2 = dst_doc.add_heading(level=2)
                    head2 = block
                    head2.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    # head2_title = head2.add_run(block.text)
                    head2.style.font.size = Pt(10)
                    head2.style.font.italic = True
                    head2.style.font.name = "Palatino Linotype"
                    head2.paragraph_format.left_indent = Cm(4.6)
                    head2.paragraph_format.line_spacing = 0.95
                    head2.paragraph_format.space_after = Pt(3)
                    head2.paragraph_format.space_before = Pt(12)
                elif "heading 3" in block.style.name.lower():
                    # head3 = dst_doc.add_heading(level=3)
                    head3 = block
                    head3.alignment = WD_ALIGN_PARAGRAPH.LEFT

                    # head3_title = head3.add_run(block.text)
                    head3.style.font.size = Pt(10)
                    head3.style.font.name = "Palatino Linotype"
                    head3.paragraph_format.left_indent = Cm(4.6)
                    head3.paragraph_format.line_spacing = 0.95
                    head3.paragraph_format.space_before = Pt(12)
                    head3.paragraph_format.space_after = Pt(3)
                else:
                    # block.style = normal_text_style
                    block.style.font.size = Pt(10)  # 样式字体大小
                    block.style.font.name = "Palatino Linotype"
                    block.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY  # 段落右对齐
                    block.paragraph_format.first_line_indent = Cm(0.75)  # 首行缩进两个字符
                    block.paragraph_format.left_indent = Cm(4.6)
                    block.paragraph_format.line_spacing = 0.95
                    block.paragraph_format.space_before = 0
                    block.paragraph_format.space_after = 0
                    for run in block.runs:
                        # if "[" in run.text:
                        #     continue
                        run.font.name = "Palatino Linotype"
                        run.font.size = Pt(10)
                    if (
                        reference_index == -1
                        and "reference" in block.text.lower()
                        and len(block.text.strip()) < 12
                    ):
                        reference_index = i
                    elif reference_index != -1:
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

                    # dst_doc.add_section(start_type=WD_SECTION.CONTINUOUS)
            else:
                block.clear()
                delete_paragraph(block)
    elif isinstance(block, Table):
        block.alignment = WD_TABLE_ALIGNMENT.CENTER
        block.style.paragraph_format.alignment = WD_TABLE_ALIGNMENT.CENTER
        block.style.font.size = Pt(10)
        rows = block.rows
        columns = block.columns
        for r in range(len(rows)):
            for c in range(len(columns)):
                cell_one = block.cell(r, c)
                for p in cell_one.paragraphs:
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for cell_run in p.runs:
                        cell_run.font.name = "Palatino Linotype"
                        cell_run.font.size = Pt(10)
    elif isinstance(block, InlineShape):
        # Read and process the Picture
        print("Picture found:", block)

dst_doc.add_section(
    start_type=WD_SECTION.CONTINUOUS
)  ###通过在页眉添加表格，在表格的单元中插入图片的方式实现在页眉插入多张图片，并两端对齐
image_header1 = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header.png"
image_header2 = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header_02.png"
sections = dst_doc.sections
# print(len(sections))
section0 = sections[0]
print("页面上边距：", section0.top_margin)
print("页面宽度：", section0.page_width)

# 设置页面距离：上下左右
section0.top_margin = Cm(2.5)
section0.bottom_margin = Cm(1.9)
section0.left_margin = Cm(1.27)
section0.right_margin = Cm(1.27)

# 设置页眉页脚的上下距离
section0.header_distance = Cm(1.8)
section0.footer_distance = Cm(0.6)
section0.different_first_page_header_footer = True
# section0.first_page_header = True
# section0.first_page_footer = True
section_0_header = section0.first_page_header
# htable = section_0_header.add_table(1, 2, Inches(6))
# htable.alignment = WD_TABLE_ALIGNMENT.CENTER
# htab_cells = htable.rows[0].cells
# ht0 = htab_cells[0].add_paragraph()
# kh = ht0.add_run()
# kh.add_picture(image_header1)

# ht1 = htab_cells[1].add_paragraph()
# ht1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
# kh1 = ht1.add_run()
# kh1.add_picture(image_header2)
from docx.enum.text import WD_ALIGN_PARAGRAPH

section_0_header.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
section0_run0 = section_0_header.paragraphs[0].add_run()
section0_run0.add_picture(image_header1, width=Cm(4.64), height=Cm(1.21))
spaces = [" "] * 141
section_0_header.paragraphs[0].add_run("".join(spaces))
section0_run1 = section_0_header.paragraphs[0].add_run()
section0_run1.add_picture(image_header2, width=Cm(1.5), height=Cm(1))

# 页脚

section_0_footer = section0.first_page_footer
htable = section_0_footer.add_table(1, 2, width=Cm(21))
htable.rows[0].height = Cm(0.5)
htab_cells = htable.rows[0].cells
# htab_cells[0].height = Cm(1)
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
# section_0_footer.paragraphs[0].style = page_header_style
# section_0_footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
# section0_run0 = section_0_footer.paragraphs[0].add_run(
#     "Int. J. Mol. Sci. 2023, 24, x. https://doi.org/10.3390/xxxxx",
# )
# section0_run0.add_tab()
# section0_run0.add_tab()
# section0_para0 = section_0_footer.paragraphs[0]
# char_num = (
#     section0.page_width
#     - len("Int. J. Mol. Sci. 2023, 24, x. https://doi.org/10.3390/xxxxx")
#     * section0_para0.style.font.size
#     - len("www.mdpi.com/journal/ijms") * section0_para0.style.font.size
# ) / section0_para0.style.font.size
# print("需要的字符数：", char_num)
# print("size:", section0_para0.style.font.size)
# # space_with = 21 - section0_para0.style.font.size * len(
# #     "Int. J. Mol. Sci. 2023, 24, x. https://doi.org/10.3390/xxxxx"
# # )

# section0_para0.add_run("www.mdpi.com/journal/ijms")


# 控制后续的页眉页脚跟前面的不一样
sections[1].header_distance = Cm(1.8)
sections[1].footer_distance = Cm(0.6)
section_1_header = sections[1].header
section_1_header.paragraphs[0].add_run("Int. J. Mol. Sci. 2023, 24, x FOR PEER REVIEW")
# section_1_htable = section_1_header.add_table(1, 2, Inches(6))
# section_1_htab_cells = section_1_htable.rows[0].cells
# section_1_ht0 = section_1_htab_cells[0].add_paragraph(
#     "Int. J. Mol. Sci. 2023, 24, x FOR PEER REVIEW"
# )
section_1_header.is_linked_to_previous = False

section_1_footer = sections[1].footer
section_1_footer.is_linked_to_previous = False

print(len(dst_doc.sections))

dst_doc.save("data/inplace_demo.docx")
