from spire.doc import *
from spire.doc.common import *

# Create word document
document = Document()

# Load a doc or docx file
document.LoadFromFile(
    "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/docx/ijms-3208768.docx"
)

# Save the document to PDF
document.SaveToFile(
    "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/output/ToPDF.pdf",
    FileFormat.PDF,
)
document.Close()
