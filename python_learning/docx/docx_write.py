from docx import Document
from docx.shared import RGBColor, Pt, Cm, Inches
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.parfmt import ParagraphFormat

document = Document()

# 设置样式
style = document.styles.add_style("textstyle", WD_STYLE_TYPE.PARAGRAPH)
style.font.size = Pt(16)  # 样式字体大小
style.font.color.rgb = RGBColor(66, 100, 0)  # 字体颜色
style.font.bold = True  # 粗体
style.font.italic = True  # 斜体
style.font.name = "Times New Roman"  # 字体类型


# 文章标题 level=0
head0 = document.add_heading(level=0)
# 标题居中
head0.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = head0.add_run(
    "Prospects and Aspirations for Workforce Training and Education in Social Prescribing",
)
title_run.font.size = Pt(24)
# 标题英文字体
title_run.font.name = "Times New Roman"
# 字体颜色
title_run.font.color.rgb = RGBColor(4, 60, 169)

# 一级标题
head1 = document.add_heading(level=1)
head1.alignment = WD_ALIGN_PARAGRAPH.LEFT
head1_title = head1.add_run("Introduction")
head1_title.font.size = Pt(24)
head1_title.font.name = "Times New Roman"
head1_title.font.color.rgb = RGBColor(4, 60, 169)

# 添加一个段落，段落添加样式
p = document.add_paragraph(
    "Social prescribing (SP) is a non-clinical intervention which involves connecting citizens to community supports to better manage their health and wellbeing and to improve outcomes [1]. SP is a means of enabling General Practitioners (GPs), nurses and other primary care professionals to refer individuals to a range of local, non-clinical services [2]. A link worker (LW) is responsible for enabling and supporting an individual to assess their requirements, co-producing solutions for them and making use of appropriate local resources [3]. Individuals who chose to use SP interventions require LWs to signpost them to various SP community activities. Taking a holistic approach in tailoring SP options to meet the requirements of individuals necessitates LWs to form relationships across a spectrum of individuals, organisations and community groups within society. LWs assist patients in managing chronic illnesses by signposting patients to community healthcare services that they were previously unaware of to improve wellbeing [4]. Tailoring patient care in SP promotes long-term wellbeing via prevention and the delaying of long-term conditions [5]. SP was also highlighted as one of the ten high impact actions outlined to ease GP workload [6]. SP intervention takes a number of different forms to seek solutions to meet an individual’s requirements. Some of the different types of SP intervention include arts-based prescription, exercise referral and green social prescribing [7,8,9]. This requires LWs to have a range of personal attributes along with good communication, knowledge and skills to navigate complex systems to develop social capital and the wellbeing of individuals and communities. As an emerging new role, LWs are not regulated by professional bodies and there is no consistent training for LWs who are joining the practice of SP from varied backgrounds. As such, LWs have varying knowledge about how to deal with individuals with complex requirements, which can impact on their decision-making capabilities to seek solutions and navigate complex systems. Therefore, this can impact on their decision-making capabilities and influence the training which may be necessary to build confidence to seek solutions [10]. Training to support LWs to manage complex case referral is essential [11]. The LW title can vary depending on the type of SP intervention; these titles include community connector, wellbeing advisor and social prescriber [12]. Although the titles can vary, there is overlap in the fundamental skills required for the role among the different titles. Recent evidence has assisted in creating internationally recognised theoretical and operational definitions of social prescribing which should be integrated into future social prescribing research and policy development [13]. There is some supporting evidence [14] indicating",
    style=style,
)


# p.alignment = WD_ALIGN_PARAGRAPH.RIGHT  # 段落右对齐
p.alignment = WD_ALIGN_PARAGRAPH.LEFT  # 段落右对齐
# p.alignment = WD_ALIGN_PARAGRAPH.CENTER  # 段落右对齐
p.paragraph_format.first_line_indent = p.style.font.size * 2  # 首行缩进两个字符


# 一级标题
head1 = document.add_heading(level=1)
head1.alignment = WD_ALIGN_PARAGRAPH.LEFT
head1_title = head1.add_run("Methods")
head1_title.font.size = Pt(24)
head1_title.font.name = "Times New Roman"
head1_title.font.color.rgb = RGBColor(4, 60, 169)

p2 = document.add_paragraph(
    "This study was funded by the Knowledge Economy Skills Scholarship (KESS 2) funded by the Welsh Government along with partner organisation Conwy Council. The purpose of this collaborative project was to examine the requirements of LWs who were delivering more SP interventions within the Conwy Council area as well as throughout Wales. The aim of this study was to explore LWs’ level of education, past and current training requirements as well as elicit LW value estimates to undertake additional training and willingness to pay (WTP) to access training."
)
p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT

# 添加一个 2×2 表格
table = document.add_table(rows=2, cols=2, style="Table Grid")
# 获取第1行第2列单元格
cell = table.cell(0, 1)

# 设置单元格文本
cell.text = "第1行第2列"

# 获取第2行
row = table.rows[1]
row.cells[0].text = "橡皮擦"
row.cells[1].text = "乔喻"

from copy import deepcopy

docx_path = "/home/bocheng/data/pdf/document_parse/Word/random/atmosphere-2489382.docx"
mydocument = Document(docx_path)
tables = mydocument.tables
new_table = deepcopy(tables[0])
para1 = document.paragraphs[3]
para1._p.addnext(new_table._element)  # 插入复制的表格


document.save("data/demo.docx")
