import pdfplumber
import fitz
import json
import re

pdf_path = "/data/bocheng/dev/mylearn/NLP-Learning/python_learning/data/pdf/online-proofreading/Online Proofreading XML Editor User Guide-Internal-v2.pdf"

import fitz  # PyMuPDF
import os


def extract_pdf_content_in_order(pdf_path, output_image_dir="images"):
    # 创建保存图片的目录
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    document = fitz.open(pdf_path)
    content_list = []  # 用于存储所有内容块（文本或图片路径）

    for page_num in range(len(document)):
        page = document.load_page(page_num)

        # 获取文本块
        text_blocks = page.get_text(
            "blocks"
        )  # [(x0, y0, x1, y1, "text", block_no, block_type), ...]

        # 获取图像块及其位置信息
        images = page.get_images(full=True)  # 获取页面中的所有图像
        image_dict = {}  # 存储图像的位置信息及对应的文件名

        for img_index, img in enumerate(images):
            xref = img[0]  # 图像的XREF编号
            base_image = document.extract_image(xref)  # 提取图像数据
            image_bytes = base_image["image"]  # 图像二进制数据
            image_ext = base_image["ext"]  # 图像格式（如png、jpg等）

            # 保存图像到本地文件
            image_filename = f"image_{page_num}_{img_index}.{image_ext}"
            image_filepath = os.path.join(output_image_dir, image_filename)
            with open(image_filepath, "wb") as image_file:
                image_file.write(image_bytes)

            # 获取图像的位置信息（边界框）
            rect = page.get_image_rects(xref)[0]  # 获取图像的矩形区域 (x0, y0, x1, y1)
            image_dict[(rect.y0, rect.x0)] = (
                image_filepath  # 使用y0, x0作为键，存储图像路径
            )

        # 合并文本块和图像块，并按位置排序
        all_blocks = []

        # 添加文本块
        for b in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b[:7]
            if block_type == 0:  # 只处理文本块
                all_blocks.append((y0, x0, f"{text.strip()}"))

        # 添加图像块
        for (y, x), image_path in image_dict.items():
            all_blocks.append((y, x, f"{os.path.abspath(image_path)}"))

        # 按照 Y 坐标（从上到下），再按 X 坐标（从左到右）排序
        for _, _, content in sorted(all_blocks, key=lambda b: (b[0], b[1])):
            if content and not re.match(r"^[0-9]{1,2}$", content.strip()):
                content_list.append(content)

    return content_list


def segment_pdf_content(content_list):
    begin_index = 0
    for i, line in enumerate(content_list):
        if re.match(r"^1\. Introduction.*[^\d]$", line.strip()):
            begin_index = i
            break
    contents = []
    all_contents = []
    for line in content_list[begin_index:]:
        if re.match(r"^[0-9]\.[0-9]\.[0-9]{1,2}.+", line.strip()):
            all_contents.append(contents)
            contents = []
        elif re.match(r"^[0-9]\.[0-9].+", line.strip()) or line.strip().startswith(
            ("1. Introduction to the Online", "2. Introduction to Online")
        ):
            all_contents.append(contents)
            contents = []
        contents.append(line)

    all_contents.append(contents)
    return all_contents


def get_table_of_contents(content_list):
    content_list=[item for  line in content_list for item in line.split("\n")]
    tab_list = []
    for i, line in enumerate(content_list):
        if re.match(r"^\d\.{10,}\d$", line.strip()):
            tab_list.append(line)
            print(line)


# 调用函数
output_image_dir = "extracted_images_xml"  # 图片保存的目录
content_list = extract_pdf_content_in_order(pdf_path, output_image_dir)
# for line in content_list:
#     print(line)
# all_contents = segment_pdf_content(content_list)

# with open("xml_editor_content.json", "w", encoding="utf-8") as f:
#     json.dump(all_contents, f, ensure_ascii=False, indent=4)
get_table_of_contents(content_list)
