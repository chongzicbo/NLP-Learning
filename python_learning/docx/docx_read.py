from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
from docx.text.paragraph import Paragraph
from docx.shape import InlineShape
import docx
import os

path = "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/docx/ijms-template.docx"
doc = docx.Document(path)
header = doc.sections[0].header


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


rels = {}
for r in doc.part.rels.values():
    if isinstance(r._target, docx.parts.image.ImagePart):
        rels[r.rId] = os.path.basename(r._target.partname)

for block in iter_block_items(doc):
    if isinstance(block, Paragraph):
        # Read and process the Paragraph
        # If you find an image
        if "Graphic" in block._p.xml:
            # Get the rId of the image
            for rId in rels:
                if rId in block._p.xml:
                    # Your image will be in os.path.join(img_path, rels[rId])
                    print("image =============================================")
        else:
            print(block.text)
    elif isinstance(block, Table):
        # Read and process the Table
        print(f"talbe was found ", block.style.name)
    elif isinstance(block, InlineShape):
        # Read and process the Picture
        print("Picture found:", block)
