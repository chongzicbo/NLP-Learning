from docx import Document
from docx.text.run import Run

from docx.shared import Inches, Pt
import docx
import os
import docx2txt

docx_path = "/home/bocheng/data/pdf/document_parse/Word/random/atmosphere-2489382.docx"
docx_path = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/test.docx"
document = Document(docx_path)
# print(help(document._element))


# def getHyperlinksRuns(paragraph):
#     def _get(node, parent):
#         for child in node:
#             if child.tag == qn("w:hyperlink"):
#                 yield from returnRun(child, parent)

#     def returnRun(node, parent):
#         for child in node:
#             if child.tag == qn("w:r"):
#                 yield Run(child, parent)

#     return list(_get(paragraph._element, paragraph))


# def getParagraphRuns(paragraph):
#     def _get(node, parent):
#         for child in node:
#             if child.tag == qn("w:r"):
#                 yield Run(child, parent)
#             if child.tag == qn("w:hyperlink"):
#                 yield from _get(child, parent)

#     return list(_get(paragraph._element, paragraph))


# for paragraph in document.paragraphs:
#     for run in paragraph.runs:
#         if not run.text.strip():
#             continue
#         print(run.text)

# for section in document.sections:
#     for paragraph in section._document_part.document.paragraphs:
#         print(paragraph.text)
# for child in document.element.body.iterchildren():
#     print(child)
# rels = {}
# for r in document.part.rels.values():
#     if isinstance(r._target, docx.parts.image.ImagePart):
#         rels[r.rId] = os.path.basename(r._target.partname)
# # Then process your text
# for paragraph in document.paragraphs:
#     # If you find an image
#     if "Graphic" in paragraph._p.xml:
#         # Get the rId of the image
#         for rId in rels:
#             if rId in paragraph._p.xml:
#                 # Your image will be in os.path.join(img_path, rels[rId])
#                 print("image =============================================")

#     else:
#         # It's not an image
#         print(paragraph.text)
# style_doc = docx.Document(
#     "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/docx/ijms-template.docx"
# )

# style_dic = {}
# for paragraph in style_doc.paragraphs:
#     # print(
#     #     paragraph.style,
#     #     "\t",
#     #     paragraph.style.name,
#     #     # "\t",
#     #     # paragraph.style.paragraph_format,
#     # )
#     style_name = paragraph.style.name.split("_")[-1]
#     if style_name == "type":
#         style_name = "article_type"
#     style_dic[style_name] = paragraph.style
# print(style_dic)
from docx.oxml.ns import qn

# for p in document.paragraphs:
#     runs = getParagraphRuns(p)
#     # runs = getHyperlinksRuns(p)
#     for run in runs:
#         run.font.name = "Palatino Linotype"
#         run.font.size = Pt(10)

# for paragraph in document.paragraphs:
#     for run in paragraph.runs:
#         run_element = run._element
#         run_parent_element = run_element.getparent()
# print(run_parent_element)
# run._element.xml = run._element.xml.replace('val="48"', 'val="64"')

# print(run._element.get("w:rPr"))
# print(run._element.items())
# hyperlink_list = run._element.findall(".//" + qn("w:hyperlink"))
# for hyperlink_item in hyperlink_list:
#     text = hyperlink_item.findall(".//" + qn("w:t"))[0].text

# print(text)
document.save("./data/hyperlink.docx")
from docx import Document
from docx import RT
import re

# # d = Document("./liu2.docx")

# for p in document.paragraphs:
#     rels = document.part.rels

#     for rel in rels:
#         if rels[rel].reltype == RT.HYPERLINK:
#             print("\n 超链接文本为", rels[rel], " 超链接网址为: ", rels[rel]._target)

for p in document.paragraphs:
    print(p.text)
