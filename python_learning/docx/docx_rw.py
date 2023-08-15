from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
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

style_dic = get_style_templates()

path = "/home/bocheng/data/pdf/document_parse/Word/random/atmosphere-2489382.docx"
src_doc = docx.Document(path)

dst_doc = docx.Document()

page_header_style = dst_doc.styles.add_style("textstyle", WD_STYLE_TYPE.PARAGRAPH)
page_header_style.font.size = Pt(8)  # 样式字体大小
page_header_style.font.color.rgb = RGBColor(66, 100, 0)  # 字体颜色

normal_text_style = dst_doc.styles.add_style(
    "normal_textstyle", WD_STYLE_TYPE.PARAGRAPH
)
normal_text_style.font.size = Pt(10)  # 样式字体大小
normal_text_style.font.name = "Palatino Linotype"


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
for block in iter_block_items(src_doc):
    if isinstance(block, Paragraph):
        # Read and process the Paragraph
        # If you find an image
        if "Graphic" in block._p.xml:
            # Get the rId of the image
            for rId in rels:
                if rId in block._p.xml:
                    # Your image will be in os.path.join(img_path, rels[rId])
                    image_path = image_name_path.get(f"image{image_id}", None)
                    if image_path is not None:
                        pic = dst_doc.add_picture(
                            open(image_path, mode="rb"),
                        )
                        pic.width = Cm(14)
                        pic.height = Cm(10)
                    image_id += 1
                    print(f"image {image_path} has been added to dst document")
        else:
            if block.text.strip():
                # style = style_dic["text"]
                # style = ParagraphStyle()
                # style.font.size = Pt(10)
                # style.font.name = "Times New Roman"  # 字体类型
                p = dst_doc.add_paragraph(block.text, style=normal_text_style)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT  # 段落右对齐
                p.paragraph_format.first_line_indent = p.style.font.size * 2  # 首行缩进两个字符
                p.paragraph_format.left_indent = Inches(1.5)
                p.paragraph_format.line_spacing = 1.5
                dst_doc.add_section(start_type=WD_SECTION.CONTINUOUS)
    elif isinstance(block, Table):
        # Read and process the Table
        para1 = dst_doc.paragraphs[-1]
        para1._p.addnext(block._element)  # 插入复制的表格
        print(f"talbe was found ", block.style.name)
    elif isinstance(block, InlineShape):
        # Read and process the Picture
        print("Picture found:", block)

###通过在页眉添加表格，在表格的单元中插入图片的方式实现在页眉插入多张图片，并两端对齐
image_header1 = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header.png"
image_header2 = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header_02.png"
sections = dst_doc.sections
print(len(sections))
section_0_header = sections[0].header
htable = section_0_header.add_table(1, 2, Inches(6))
htab_cells = htable.rows[0].cells
ht0 = htab_cells[0].add_paragraph()
kh = ht0.add_run()
kh.add_picture(image_header1)

ht1 = htab_cells[1].add_paragraph()
ht1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
kh1 = ht1.add_run()
kh1.add_picture(image_header2)


# 页脚

section_0_footer = sections[0].footer
htable = section_0_footer.add_table(1, 2, Inches(6))
htab_cells = htable.rows[0].cells
ht0 = htab_cells[0].add_paragraph(
    "Int. J. Mol. Sci. 2023, 24, x. https://doi.org/10.3390/xxxxx",
    style=page_header_style,
)
ht0.alignment = WD_ALIGN_PARAGRAPH.LEFT

ht1 = htab_cells[1].add_paragraph("www.mdpi.com/journal/ijms")
ht1.alignment = WD_ALIGN_PARAGRAPH.RIGHT


# para = section_0_header.add_paragraph()
# para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY_MED
# run = para.add_run()
# run.add_picture(
#     open(
#         "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header.png",
#         mode="rb",
#     )
# )
# run.add_text("\t\t\t\t  ")

# # para1 = section_0_header.add_paragraph()
# # para1.alignment = WD_ALIGN_PARAGRAPH.LEFT
# run = para.add_run("")
# run.add_picture(
#     open(
#         "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/pageheader/page_header.png",
#         mode="rb",
#     )
# )


section_1_header = sections[1].header
section_1_header.is_linked_to_previous = False
htable = section_1_header.add_table(1, 2, Inches(6))
htab_cells = htable.rows[0].cells
ht0 = htab_cells[0].add_paragraph(
    "Int. J. Mol. Sci. 2023, 24, x FOR PEER REVIEW", style=page_header_style
)

section_1_footer = sections[1].footer
section_1_footer.is_linked_to_previous = False
dst_doc.save("data/demo.docx")
