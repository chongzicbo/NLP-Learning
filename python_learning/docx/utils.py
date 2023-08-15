import docx

style_doc = docx.Document(
    "/home/bocheng/dev/mylearn/NLP-Learning/python_learning/docx/data/docx/ijms-template.docx"
)


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
