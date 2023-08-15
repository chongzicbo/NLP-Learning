from docx import Document
from docx.shared import Inches
import docx
import os
import docx2txt

# docx_path = "/home/bocheng/data/pdf/document_parse/Word/random/atmosphere-2489382.docx"
# docx2txt.process(docx_path, "data/")
# document = Document(docx_path)
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
style_doc = docx.Document(
    "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/docx/ijms-template.docx"
)

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
print(style_dic)
